"""
This module defines viewers for `circuits.Unit` and `circuits.Connector`.
"""
import typing as T

from circuits import connector, units
from circuits.rendering import cells, colors, paths, renderer, viewers

COLORS = colors.COLORS

DirectionSpec = T.Literal['right', 'down', 'left', 'up', 'r', 'd', 'l', 'u', 'x+', 'x-', 'y+', 'y-']
"""User-specified metadata options for a waypoint direction (relative).

Options: absolute direction ('right'/'r', 'down'/'d', 'left'/'l', 'up'/'u'), outward ('x+') or
inward ('x-') along the parent x-axis, outward ('y+') or inward ('y-') along the parent y-axis.
"""


class UnitMeta(viewers.SignalMeta, total=False):
  """Metadata for `circuits.Unit`."""
  collapse: bool
  """Whether to collapse the unit as a single output cell without input connection cells."""
  # TODO: Implement collapse.


class ConnectionMeta(T.TypedDict, total=False):
  """Metadata for `circuits.Connection`."""
  visible: bool
  """Whether to render."""

  color: str | T.Literal['sign', 'rainbow', 'cycle']
  """Color string or coloring scheme for connection cells."""

  sign: T.Literal['+', '-']
  """Connection sign, for coloring scheme."""

  waypoints: T.Iterable[tuple[float, float, DirectionSpec]]
  """Connection path waypoints, with positions/directions relative to the signal viewer's coordinate
  system, which is defined by an x-axis pointing in the flow direction and a y-axis pointing along
  the growth direction away from the anchor point. Using this relative coordinate system makes it
  easy to specify circuit layouts with left/right and up/down symmetries, since changing the signal
  viewer's `flow` or `anchor` metadata will flip the waypoints appropriately for a mirrored layout.
  """


def waypoints_abs(
  meta: ConnectionMeta,
  *,
  xy_abs: tuple[int, int],
  flow_abs: viewers.FlowAbs,
  anchor_abs: viewers.AnchorAbs,
) -> list[tuple[int, int, paths.Direction]]:
  """Convert waypoint specifications into absolute coordinates and directions."""
  x_abs, y_abs = xy_abs
  xaxis_abs = +1 if flow_abs == 'right' else -1
  yaxis_abs = +1 if anchor_abs == 'bottom' else -1
  waypoints = []
  for waypoint in meta.get('waypoints', []):
    wx, wy, wdirstr = waypoint
    wx_abs = x_abs + xaxis_abs * wx
    wy_abs = y_abs + yaxis_abs * wy
    match wdirstr:
      case 'right' | 'r':
        wdir = paths.Direction.RIGHT
      case 'down' | 'd':
        wdir = paths.Direction.DOWN
      case 'left' | 'l':
        wdir = paths.Direction.LEFT
      case 'up' | 'u':
        wdir = paths.Direction.UP
      case 'x+':
        wdir = paths.Direction.RIGHT if xaxis_abs * wx >= 0 else paths.Direction.LEFT
      case 'x-':
        wdir = paths.Direction.LEFT if xaxis_abs * wx >= 0 else paths.Direction.RIGHT
      case 'y+':
        wdir = paths.Direction.UP if yaxis_abs * wy >= 0 else paths.Direction.DOWN
      case 'y-':
        wdir = paths.Direction.DOWN if yaxis_abs * wy >= 0 else paths.Direction.UP
      case _:
        raise ValueError(f'Invalid waypoint direction: {wdirstr}')
    waypoints.append((wx_abs, wy_abs, wdir))
  return waypoints


class UnitViewer(viewers.SignalViewer):
  """Viewer for `circuits.Unit`."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    node = T.cast(units.Basic | units.Adaptor | units.Oscillator, self.node)
    meta = T.cast(UnitMeta, self.node.meta)
    cfg = self.renderer.cfg

    if not meta.get('visible', True): return
    xy_abs = meta['xy_abs']
    x, y = xy_abs
    flow = T.cast(viewers.FlowAbs, self.flow)
    anchor = T.cast(viewers.AnchorAbs, self.anchor)

    # Initialize divider.
    self._divider.set(
      ydata=((y, y + max(1, len(node.synapses))) if anchor == 'bottom' else
             (y, y - max(1, len(node.synapses)))),
    )

    # Initialize synapses (cells and paths).
    self._synapses: dict[str, tuple[connector.Connection, list[cells.SignalCell], T.Any]] = {}
    for dy, (name, synapse) in enumerate(node.synapses.connections.items()):
      source_meta = T.cast(viewers.SignalMeta, synapse.source.meta)
      source_x, source_y = source_meta['xy_abs']
      source_flow = viewers.flow_abs(cfg, source_meta)
      source_anchor = viewers.anchor_abs(cfg, source_meta)

      synapse_meta = T.cast(ConnectionMeta, synapse.meta)
      synapse_sign = synapse_meta.get('sign', None)
      synapse_color = synapse_meta.get('color', 'sign')
      match synapse_color:
        case 'sign':
          match synapse_sign:
            case '+':
              synapse_linecolor = COLORS.red
              synapse_bgcolor = COLORS.redl
            case '-':
              synapse_linecolor = COLORS.blue
              synapse_bgcolor = COLORS.bluel
            case _:
              synapse_linecolor = COLORS.gray
              synapse_bgcolor = COLORS.grayll
        case 'rainbow':
          raise NotImplementedError
        case 'cycle':
          raise NotImplementedError
        case _:
          synapse_linecolor = synapse_color
          synapse_bgcolor = COLORS.grayll

      synapse_arrow: dict[str, T.Any]
      match synapse_sign:
        case '+':
          synapse_arrow = dict(arrow_style='->')
        case '-':
          synapse_arrow = dict(arrow_style='|-|', arrow_kwargs=dict(widthA=0, widthB=0.35))
        case _:
          synapse_arrow = dict(arrow_style='-')

      synapse_cells = []  # Ordered [t-1, t-2, ..., t-delay].
      cy = y + dy if anchor == 'bottom' else y - dy - 1
      for dx in range(len(synapse.delay)):
        cx = x - dx - 1 if flow == 'right' else x + dx
        synapse_cells.append(
          cells.SignalCell(
            self.ax,
            cx,
            cy,
            label=cfg.label_inputs_values,
            bgcolor=synapse_bgcolor,
          )
        )

      synapse_path = paths.OrthogonalPath(
        self.ax,
        start=(
          source_x if source_flow == 'right' else source_x - 1,
          source_y if source_anchor == 'bottom' else source_y - 1,
        ),
        end=(cx, cy),  # type: ignore  # Use frontmost cell (last to be drawn).
        start_side=paths.Side.EAST if source_flow == 'right' else paths.Side.WEST,
        end_side=paths.Side.WEST if flow == 'right' else paths.Side.EAST,
        waypoints=waypoints_abs(synapse_meta, xy_abs=xy_abs, flow_abs=flow, anchor_abs=anchor),
        path_nudge=0.1 * (dy + 1) * (1 if anchor == 'bottom' else -1),
        linecolor=synapse_linecolor,
        **synapse_arrow,
      ) if synapse_meta.get('visible', True) else None
      self._synapses[name] = (synapse, synapse_cells, synapse_path)

  def draw(self):
    meta = T.cast(UnitMeta, self.node.meta)
    if not meta.get('visible', True): return []

    artists = list(super().draw())

    # Draw synapse cells.
    for name, (synapse, synapse_cells, synapse_path) in self._synapses.items():
      t = 0
      for t, cell in enumerate(synapse_cells):
        mean = synapse.delay[-t - 1].mean().item()
        cell.update(mean)
        artists.extend(cell.draw())
      synapse_path.update(synapse.delay[-t - 1].mean().item())
      artists.extend(synapse_path.draw())
    return artists


renderer.register(units.Basic, UnitViewer)
renderer.register(units.Adaptor, UnitViewer)
renderer.register(units.Oscillator, UnitViewer)

__all__ = [
  'UnitMeta', 'ConnectionMeta', 'UnitViewer',
  'DirectionSpec', 'waypoints_abs',
] # yapf: disable
