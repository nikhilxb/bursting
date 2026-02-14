"""
This module defines viewers for `circuits.Monitor`.
"""
import typing as T

import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np

from circuits import monitors, signal
from circuits.rendering import colors, paths, renderer, viewers

COLORS = colors.COLORS


class TimeseriesMonitorSourceMeta(T.TypedDict, total=False):
  """Metadata for a `circuits.TimeseriesMonitorSource`."""
  color: str
  """Color string for the plot line."""


class TimeseriesMonitorMeta(renderer.RenderedNodeMeta, total=False):
  """Metadata for `circuits.TimeseriesMonitor`."""
  visible: bool
  """Whether to render."""

  width: int
  """Width of the monitor in cells."""

  height: int
  """Height of the monitor in cells."""

  ylim: tuple[float | None, float | None]
  """Y-axis limits. If None, the y-axis autoscales."""

  xlim: tuple[float | None, float | None]
  """X-axis limits. If None, the x-axis autoscales."""

  xwindow: float | None
  """Sliding window size along x-axis. If a float, the x-axis range is a fixed window that shifts
  every `xwindow` timesteps, which is efficient for rendering. If `None`, the x-axis range grows
  with time, changing the horizontal scaling of data and requiring an axis redraw every timestep."""
  # TODO: Implement xwindow.


class TimeseriesMonitorViewer(renderer.Viewer):
  """Viewer for `circuits.TimeseriesMonitor`."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    node = T.cast(monitors.TimeseriesMonitor, self.node)
    meta = T.cast(TimeseriesMonitorMeta, self.node.meta)
    cfg = self.renderer.cfg

    if not meta.get('visible', True): return
    x, y = meta['xy_abs']
    w = meta.get('width', 6)
    h = meta.get('height', 4)
    name = meta.get('name', '')

    # Create a subaxis to plot on.
    self._sax = sax = self.ax.inset_axes([x, y, w, h], transform=self.ax.transData)

    self._connectors: dict[signal.Signal, paths.OrthogonalPath] = {}
    self._lines: list[mlines.Line2D] = []
    for i, source in enumerate(node.sources):
      source_meta = T.cast(TimeseriesMonitorSourceMeta, source.meta)
      source_color = source_meta.get('color', f'C{(i+2)%10}')
      self._lines.append(sax.plot([], [], lw=1, c=source_color)[0])
      if source.signal not in self._connectors:
        signal_meta = T.cast(viewers.SignalMeta, source.signal.meta)
        signal_x, signal_y = signal_meta['xy_abs']
        signal_flow = viewers.flow_abs(cfg, signal_meta)
        signal_anchor = viewers.anchor_abs(cfg, signal_meta)

        self._connectors[source.signal] = paths.OrthogonalPath(
          self.ax,
          start=(x, y),
          start_size=(w, h),
          end=(
            signal_x if signal_flow == 'right' else signal_x - 1,
            signal_y if signal_anchor == 'bottom' else signal_y - 1,
          ),
          start_side=paths.Side.EAST if x + w / 2 <= signal_x else paths.Side.WEST,
          end_side=paths.Side.SOUTH if y + h / 2 <= signal_y else paths.Side.NORTH,
          arrow_style='-',
          linewidth=0.5,
          linecolor=COLORS.black,
        )
    sax.set(
      xlim=meta.get('xlim', (0, None)),
      ylim=meta.get('ylim', None),
      title=f'{name}',
      facecolor=COLORS.white,
    )
    sax.spines[:].set_visible(True)
    sax.xaxis.set_minor_locator(mticker.MultipleLocator(1000))  # Minor tick every 1000 ms.

    # Maintain y-axis limits based on data seen so far, so that the y-axis doesn't jump around.
    self._ymin = None
    self._ymax = None

  def draw(self):
    node = T.cast(monitors.TimeseriesMonitor, self.node)
    meta = T.cast(TimeseriesMonitorMeta, self.node.meta)
    if not meta.get('visible', True): return []

    for i, source in enumerate(node.sources):
      xdata = self.renderer.sim.time - node.dt * np.arange(len(source.buffer))[::-1]
      ydata = np.array([tensor.mean().item() for tensor in source.buffer])
      if len(ydata) > 0:
        self._ymin = ydata.min() if self._ymin is None else min(self._ymin, ydata.min())
        self._ymax = ydata.max() if self._ymax is None else max(self._ymax, ydata.max())
      self._lines[i].set_data(xdata, ydata)

    sax = self._sax
    ymin, ymax = meta.get('ylim', (None, None))
    sax.set(
      xlim=(0, self.renderer.sim.time),
      ylim=(ymin if ymin is not None else self._ymin, ymax if ymax is not None else self._ymax),
    )

    return self._lines + [sax.xaxis, sax.yaxis]


renderer.register(monitors.TimeseriesMonitor, TimeseriesMonitorViewer)

__all__ = ['TimeseriesMonitorMeta', 'TimeseriesMonitorViewer']
