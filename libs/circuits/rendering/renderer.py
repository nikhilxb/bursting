"""
This module defines the `Renderer` class, which visualizes a circuit's simulation state. It is
built on top of `matplotlib`'s low-level APIs that draw shapes (i.e. artists, patches, paths) on a
canvas.

The `Renderer` utilizes 2 abstractions to decompose and decouple the rendering implementation:

1. The `Drawable` interface defines general-purpose visual elements (e.g. grid cells, orthogonal
paths) that can be rendered on the canvas's grid-based layout. They do not know about the
`circuits` library.

2. The `Viewer` interface defines specialized visualizations that are associated with specific
types of `circuits.Node` (e.g. units, generators, monitors). They are often composed from multiple
`Drawable` elements, and they know how to interpret their underlying node's internal state.
"""
import abc
import dataclasses
import os
import typing as T

import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np

from circuits import nodes, simulator
from circuits.rendering import colors

COLORS = colors.COLORS


@dataclasses.dataclass
class RendererConfig:
  dpi: int = 100
  """Number of pixels in 1 inch. Default: 100"""

  grid_size: int = 20
  """Number of pixels width/height of each grid square."""

  grid_major: int = 10
  """Number of squares separating major grid lines."""

  grid_minor: int = 1
  """Number of squares separating minor grid lines."""

  grid_pad: int = 5
  """Number of squares to extend autoscaled viewport limits."""

  grid_lines: bool = False
  """Show major/minor grid lines."""

  grid_coords: bool = False
  """Show major grid coordinate labels."""

  label_outputs: bool = True
  """Show `Signal` names at output cells."""

  label_outputs_values: bool = False
  """Show `Signal` values at output cells."""

  label_inputs: bool = False
  """Show `Signal` names at input cells."""

  label_inputs_values: bool = False
  """Show `Signal` values at input cells."""

  label_timestamp: bool = False
  """Show `Simulator` time/steps."""

  layout_flow: T.Literal['right', 'left', 'out', 'in'] = 'out'
  """Direction of input-to-output flow. If 'out', flows towards 'right' for positive x and towards
  'left' for negative x."""

  layout_anchor: T.Literal['top', 'bottom', 'out', 'in'] = 'in'
  """Location of a component's anchor point that is positioned by `xy` metadata. If 'in', anchor
  at 'bottom' for positive y and anchor at 'top' for negative y."""


class Drawable(abc.ABC):
  """A general-purpose visual element that can be rendered on the canvas."""
  @abc.abstractmethod
  def draw(self) -> T.Iterable[martist.Artist]:
    """Modifies artists to visualize any updated state.
    Returns:
      Artists that were modified and need to be re-rendered.
    """
    pass


class Viewer(Drawable):
  """A specialized visualization that is associated with a specific type of `circuits.Node`.

  Attributes:
    node: Node associated with this viewer.
    fig: Figure where this viewer is rendered.
    ax: Axes where this viewer is rendered.
    renderer: Renderer that manages this viewer.
  """
  def __init__(
    self,
    *,
    node: nodes.Node,
    fig: mfigure.Figure,
    ax: maxes.Axes,
    renderer: 'Renderer',
  ):
    # Required keyword arguments.
    self.node = node
    self.fig = fig
    self.ax = ax
    self.renderer = renderer


try:
  VIEWERS  # type: ignore
except NameError:
  # Initialize only if not already defined.
  VIEWERS: dict[T.Type[nodes.Node], T.Type[Viewer]] = {}
  """Registry mapping `Node` subclasses to `Viewer` subclasses."""


def register(node_cls: T.Type[nodes.Node], viewer_cls: T.Type[Viewer]):
  """Registers the `Node` subclass to be visualized by the `Viewer` subclass."""
  VIEWERS[node_cls] = viewer_cls


def _autoscale_lim(
  lim: int | tuple[int | None, int | None] | None,
  vmin: int,
  vmax: int,
  *,
  pad: int = 0,
) -> tuple[int, int]:
  """Calculates limits of a viewport axis based on user constraints and/or content bounds.

  Args:
    lim: User constraints. If `None`, limits are autoscaled to fit content. If a 2-tuple, limits are
      constrained to the specified lower/upper int values and autoscaled otherwise. If a single
      int, limits are constrained to `(0, lim)` if positive and `(lim, 0)` if negative.
    vmin: Content lower bound.
    vmax: Content upper bound.
    pad: Padding to extend autoscaled limits beyond content bounds (must be >= 0).

  Returns:
    Limits of viewport axis (lower, upper).
  """
  if not lim:
    lower = vmin - pad
    upper = vmax + pad
  elif isinstance(lim, int):
    lower, upper = (0, lim) if lim > 0 else (lim, 0)
  else:
    lower = lim[0] if lim[0] is not None else vmin - pad
    upper = lim[1] if lim[1] is not None else vmax + pad
  if lower >= upper: raise ValueError(f"Invalid limits: given {lim}, calculated {(lower, upper)}")
  return (lower, upper)


class RenderedNodeMeta(T.TypedDict):
  name: str
  """Name based on its attachment on the parent node."""

  xy: tuple[int, int]
  """Relative xy-position to the parent xy-position."""

  xy_abs: tuple[int, int]
  """Absolute xy-position (relative to grid origin)."""

  xy_abs_parent: tuple[int, int]
  """Absolute xy-position of the parent node (relative to grid origin)."""


class Renderer:
  # TODO: Viewer bounding box calculations with a bottom-up traversal of the circuit tree. This
  #   enables accurate viewport limits. It also enables container viewers (e.g. `Group` nodes) to
  #   resize themselves based on children and render a padded bounding box.
  # TODO: Conditional child viewer rendering controlled by parent viewer. This enables parent
  #   viewers to hide/show children based on their own metadata (e.g. `visible`).

  def __init__(
    self,
    sim: simulator.Simulator,
    *,
    xlim: int | tuple[int | None, int | None] | None = None,
    ylim: int | tuple[int | None, int | None] | None = None,
    **kwargs,
  ):
    """
    Args:
      sim: Top-level `Simulator` node to visualize.
      xlim,ylim: Limits of viewport axes (in grid squares). If `None`, limits are autoscaled to fit
        content. If a 2-tuple, limits are constrained to the specified lower/upper int values and
        autoscaled otherwise. If a single int, limits are constrained to `(0, lim)` if positive and
        `(lim, 0)` if negative. Default: `None`.
      **kwargs: Override key-values in `RendererConfig`.
    """
    self.sim = sim
    self.cfg = cfg = RendererConfig(**kwargs)

    # Initialize node metadata.
    ctx = self._init_nodes(self.sim)

    # Set up `Figure` and `Axes` with grid coordinates, with positive-x pointing rightwards and
    # positive-y pointing upwards, as in a standard Cartesian coordinate system.
    #
    # The content bounds (xmin, xmax, ymin, ymax) are calculated from `node.meta.xy_abs` anchor
    # points, so do not factor in viewer bounding boxes. Getting such bounding boxes would be
    # difficult, as they need to be dynamically calculated with a bottom-up traversal of the
    # circuit tree, but viewers need to be created with `fig`/`ax` references, which depend on
    # autoscaled viewport limits, so there would be a circular dependency. Instead, we rely on
    # a workaround where autoscaled viewport limits are padded by a fixed amount (`cfg.grid_pad`),
    # and users can manually constrain viewport limits with `xlim`/`ylim` arguments.
    self.xlim = xlim = _autoscale_lim(xlim, ctx['xmin'], ctx['xmax'], pad=cfg.grid_pad)
    self.ylim = ylim = _autoscale_lim(ylim, ctx['ymin'], ctx['ymax'], pad=cfg.grid_pad)
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    self.fig = fig = plt.figure(
      figsize=(width * cfg.grid_size / cfg.dpi, height * cfg.grid_size / cfg.dpi),
      dpi=cfg.dpi,
      facecolor='white',  # Needed to prevent aliasing issues with lines and text.
      edgecolor='white',
      linewidth=2,
      zorder=10,
    )
    self.ax = ax = fig.add_axes((0, 0, 1, 1), aspect='equal', frameon=False)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Plot grid lines (and optionally coordinate labels).
    if cfg.grid_lines:
      ax.tick_params(
        which='both',
        left=False,
        right=False,
        top=False,
        bottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=False,
      )
      # Skip first/last gridlines, as lines on the figure border are asymmetrically rendered and
      # coordinate labels would be clipped.
      xticks_major = list(range(xlim[0] + cfg.grid_major, xlim[1], cfg.grid_major))
      yticks_major = list(range(ylim[0] + cfg.grid_major, ylim[1], cfg.grid_major))
      ax.set_xticks(xticks_major)
      ax.set_yticks(yticks_major)
      ax.set_xticks(range(xlim[0] + cfg.grid_minor, xlim[1], cfg.grid_minor), minor=True)
      ax.set_yticks(range(ylim[0] + cfg.grid_minor, ylim[1], cfg.grid_minor), minor=True)
      ax.grid(which='major', color=COLORS.gray, lw=0.5, alpha=1)
      ax.grid(which='minor', color=COLORS.grayll, lw=0.5, alpha=1)
      ax.axhline(0, color=COLORS.graydd, lw=0.5)
      ax.axvline(0, color=COLORS.graydd, lw=0.5)
      ax.set_axisbelow(True)

      # Plot coordinate labels at grid intersections.
      if cfg.grid_coords:
        for x in xticks_major:
          for y in yticks_major:
            ax.text(
              x,
              y,
              f'({x},{y})',
              c=COLORS.graydd,
              size='medium',
              va='center',
              ha='center',
              zorder=1,
            )

    # Plot frame timestamp.
    self._timestamp = ax.annotate(
      '',
      xy=(0, 1),
      xycoords='axes fraction',
      c=COLORS.graydd,
      size='medium',
      ha='left',
      va='top',
      zorder=10,
    ) if cfg.label_timestamp else None

    # Do not automatically show; frames can be generated with `render()`.
    plt.close()

    # Initialize `Viewer` instances for all `Simulator` nodes.
    self._viewers: dict[nodes.ModuleNode, Viewer] = {}
    self._init_viewers(self.sim)

  def _init_nodes(
    self,
    node: nodes.ModuleNode,
    *,
    xy_abs: tuple[int, int] = (0, 0),
    ctx: dict[str, T.Any] | None = None,
  ):
    """Initializes node metadata by traversing the circuit tree.

    Add to metadata fields (`node.meta`):
      (1) `name` (str): Node name from attachment on parent.
      (2) `xy` (tuple[int, int]): Relative xy-position to parent, if not already exists.
      (3) `xy_abs` (tuple[int, int]): Absolute xy-position relative to grid origin, calculated with
        a running sum of xy-positions relative to ancestors.
      (4) `xy_abs_parent` (tuple[int, int]): Absolute xy-position of parent node.

    Propagate context down the tree:
      (1) `xmin`/`xmax`/`ymin`/`ymax` (int): Viewport limits, calculated with a running min/max of
        content bounds.

    Args:
      node: Current node in traversal.
      xy_abs: Absolute xy-position relative to grid origin.
      ctx: Modifiable context propagating down the tree.

    Returns:
      ctx: Modified context propagating up the tree.
    """
    # Set relative/absolute xy-positions.
    node.meta['xy_abs_parent'] = xy_abs
    xy = node.meta.setdefault('xy', (0, 0))
    xy_abs = (xy_abs[0] + xy[0], xy_abs[1] + xy[1])
    node.meta['xy_abs'] = xy_abs

    # Update context.
    if ctx is None:
      ctx = dict(
        xmin=xy_abs[0],
        xmax=xy_abs[0],
        ymin=xy_abs[1],
        ymax=xy_abs[1],
      )
    else:
      ctx['xmin'] = min(xy_abs[0], ctx['xmin'])
      ctx['xmax'] = max(xy_abs[0], ctx['xmax'])
      ctx['ymin'] = min(xy_abs[1], ctx['ymin'])
      ctx['ymax'] = max(xy_abs[1], ctx['ymax'])

    if isinstance(node, nodes.PrimitiveNode): return ctx

    for name, child in node.items():
      if isinstance(child, (nodes.PrimitiveNode, nodes.ContainerNode)):
        # Set child node name from attachment on parent.
        child.meta['name'] = name
        self._init_nodes(child, xy_abs=xy_abs, ctx=ctx)

    return ctx

  def _init_viewers(self, node: nodes.ModuleNode):
    """
    Initializes registered viewers for nodes by traversing the circuit tree.

    Args:
      node: Current node in traversal.
    """
    # Get most specific superclass of `node` for which a `Viewer` is registered.
    viewer_cls = next((VIEWERS[ncls] for ncls in type(node).__mro__ if ncls in VIEWERS), None)
    if viewer_cls is not None:
      # Adds viewer in correct order for hierarchical rendering.
      self._viewers[node] = viewer_cls(node=node, fig=self.fig, ax=self.ax, renderer=self)

    if isinstance(node, nodes.PrimitiveNode): return
    for child in node.children():
      if isinstance(child, (nodes.PrimitiveNode, nodes.ContainerNode)):
        self._init_viewers(child)

  def render(self):
    """Renders the canvas reflecting the current simulation state to a numpy array.

    Returns:
      pixels: Numpy array of the canvas RGBA buffer, shape (rows, cols, 4).
    """
    self._render()
    # Save pixels of updated figure into numpy array.
    self.fig.canvas.draw()
    return np.array(self.fig.canvas.buffer_rgba())  # type: ignore

  def render_to_file(self, path: str | os.PathLike, **kwargs):
    """Renders the canvas reflecting the current simulation state to a file."""
    self._render()
    # Set `transparent=False` in order to render `figure.facecolor` properly (as white), which is
    # needed to prevent aliasing issues with lines and text.
    self.fig.savefig(
      str(path), **{
        'dpi': self.cfg.dpi, 'transparent': False, 'pad_inches': 0, **kwargs
      }
    )

  def _render(self):
    """Renders the canvas reflecting the current simulation state."""
    for viewer in self._viewers.values():
      # TODO: Use returned artists to optimize rendering via blitting.
      viewer.draw()

    if self._timestamp:
      # TODO: Use a format string from `RendererConfig`.
      self._timestamp.set_text(f'Step = {self.sim.steps} | Time = {self.sim.time:.1f} ms')


__all__ = ['Drawable', 'Viewer', 'Renderer', 'RendererConfig', 'register']
