"""
This module defines viewers for `circuits.Signal`.
"""
import typing as T

import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.transforms as mtransforms

from circuits import signal
from circuits.rendering import cells, colors, renderer

COLORS = colors.COLORS

FlowSpec = T.Literal['right', 'left', 'out', 'in', 'r', 'l', 'x+', 'x-']
"""User-specified metadata options for flow direction (relative), which is the horizontal direction
that signal flows out of the signal viewer.

Options: absolute direction ('right'/'r', 'left'/'l'), flowng outward ('out'/'x+') or inward
('in'/'x-') along the parent x-axis.
"""

FlowAbs = T.Literal['right', 'left']
"""Calculated metadata options for flow direction (absolute)."""

AnchorSpec = T.Literal['top', 'bottom', 'out', 'in', 't', 'b', 'y+', 'y-']
"""User-specified metadata options for anchor placement (relative), which is the side of the signal
viewer that is fixed in vertical position, while the opposite side can grow to accomodate
additional visual elements.

Options: absolute placement ('top'/'t', 'bottom'/'b'), anchor outside ('out'/'y+') or inside
('in'/'y-') along the parent y-axis.
"""

AnchorAbs = T.Literal['top', 'bottom']
"""Calculated metadata options for anchor placement (absolute)."""

DecoratorSpec = T.Literal['>', 'o>', ')', 'o)', '~']
"""User-specified metadata options for the decorator, which indicates the signal flow direction and
can semantically group different signal cell types.

Options: solid/outlined triangle ('>', 'o>'), solid/outlined circle (')', 'o)'), wave `~`.
"""


class SignalMeta(renderer.RenderedNodeMeta, total=False):
  """Metadata for `circuits.Signal`."""
  visible: bool
  """Whether to render."""

  color: str
  """Color string for output cell decorator."""

  flow: FlowSpec
  """Horizontal flow direction. See `FlowSpec`."""

  anchor: AnchorSpec
  """Vertical anchor placement. See `AnchorSpec`."""

  decorator: DecoratorSpec
  """Shape of output cell decorator. See `DecoratorSpec`."""


def flow_abs(cfg: renderer.RendererConfig, meta: SignalMeta) -> FlowAbs:
  """Convert flow direction specification into absolute flow direction."""
  flow = meta.get('flow', cfg.layout_flow)
  x, _ = meta['xy']  # Relative position to parent.
  match flow:
    case 'right' | 'r':
      return 'right'
    case 'left' | 'l':
      return 'left'
    case 'out' | 'x+':
      return 'right' if x >= 0 else 'left'
    case 'in' | 'x-':
      return 'left' if x >= 0 else 'right'
  raise ValueError(f'Invalid meta field: {flow = }')


def anchor_abs(cfg: renderer.RendererConfig, meta: SignalMeta) -> AnchorAbs:
  """Convert anchor placement specification into absolute anchor placement."""
  anchor = meta.get('anchor', cfg.layout_anchor)
  _, y = meta['xy']  # Relative position to parent.
  match anchor:
    case 'top' | 't':
      return 'top'
    case 'bottom' | 'b':
      return 'bottom'
    case 'out' | 'y+':
      return 'top' if y >= 0 else 'bottom'
    case 'in' | 'y-':
      return 'bottom' if y >= 0 else 'top'
  raise ValueError(f'Invalid meta field: {anchor = }')


class SignalViewer(renderer.Viewer):
  """Viewer for `circuits.Signal`."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    meta = T.cast(SignalMeta, self.node.meta)
    cfg = self.renderer.cfg

    if not meta.get('visible', True): return
    x, y = meta['xy_abs']
    self.flow = flow = flow_abs(cfg, meta)
    self.anchor = anchor = anchor_abs(cfg, meta)
    color = meta.get('color', COLORS.black)
    decorator = meta.get('decorator', '>')

    # Initialize output cell.
    self._output = cells.SignalCell(
      self.ax,
      x if flow == 'right' else x - 1,
      y if anchor == 'bottom' else y - 1,
      label=cfg.label_outputs_values,
    )

    # Initialize decorator.
    match decorator:
      case '>' | 'o>':
        decorator = dict(
          marker=9 if flow == 'right' else 8,
          markersize=5,
          markerfacecolor=color if decorator == '>' else 'white',
          markeredgecolor=color,
          markeredgewidth=1,
        )
      case ')' | 'o)':
        decorator = dict(
          marker=mmarkers.MarkerStyle(
            '$\u25D7$' if flow == 'right' else '$\u25D6$',
            transform=mtransforms.Affine2D().translate(0.2 if flow == 'right' else -0.4, 0),
          ),
          markersize=5,
          markerfacecolor=color if decorator == ')' else 'white',
          markeredgecolor=color,
          markeredgewidth=1,
        )
      case '~':
        decorator = dict(
          marker=mmarkers.MarkerStyle(
            '$\u223F$',
            transform=(
              mtransforms.Affine2D().scale(1.2, 1.2).translate(0.5 if flow == 'right' else -0.7, 0)
            ),
          ),
          markersize=5,
          markerfacecolor=color,
          markeredgecolor=color,
          markeredgewidth=1,
        )
      case _:
        raise ValueError(f'Invalid decorator: {decorator}')
    self._decorator = self.ax.plot(x, y + 0.5 if anchor == 'bottom' else y - 0.5, **decorator)

    # Initialize divider.
    self._divider = self.ax.add_line(
      mlines.Line2D(
        (x, x),
        (y, y + 1) if anchor == 'bottom' else (y, y - 1),
        lw=1.,
        c=color,
      )
    )

    # Initialize output cell label.
    if cfg.label_outputs:
      self._text = self.ax.annotate(
        f"{meta['name']}",
        xy=(x, y),
        textcoords='offset points',
        xytext=(0, -3 if anchor == 'bottom' else 2),
        ha='center',
        va='top' if anchor == 'bottom' else 'bottom',
        c=COLORS.graydd,
        size='medium',
      )

  def draw(self):
    node = T.cast(signal.Signal, self.node)
    meta = T.cast(SignalMeta, self.node.meta)
    if not meta.get('visible', True): return []

    mean = node.value.mean().item()
    self._output.update(mean)
    return self._output.draw()


renderer.register(signal.Signal, SignalViewer)

__all__ = [
  'SignalMeta', 'SignalViewer',
  'FlowSpec', 'FlowAbs', 'flow_abs',
  'AnchorSpec', 'AnchorAbs', 'anchor_abs',
] # yapf: disable
