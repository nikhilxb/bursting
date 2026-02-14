import collections
import collections.abc
import dataclasses
import typing as T

import hsluv
import matplotlib.axes as maxes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Aspect ratio of height/width.
RATIOS = dict(
  square=1,
  standard=3 / 4,
  half=1 / 2,
  third=1 / 3,
  fourth=1 / 4,
  fifth=1 / 5,
  sixth=1 / 6,
)


def figsize(
  scale: float = 1.,
  aspect: float | None = None,
  r: int = 1,
  c: int = 1,
  *,
  w: float | None = None,
  h: float | None = None,
  dw: float = 0.,
  dh: float = 0.,
) -> np.ndarray:
  """Calculate figure size based on scale, aspect ratio, rows, and columns.

  Args:
    scale: Scale factor for default size.
    aspect: Aspect ratio of height/width.
    r: Number of rows.
    c: Number of columns.
    w: Absolute width (in inches).
    h: Absolute height (in inches).
    dw: Delta width (in inches).
    dh: Delta height (in inches).
  """
  kw, kh = plt.rcParams['figure.figsize']
  # Set width as: (1) absolute width [inches], (2) scaled default.
  width = w or scale * kw
  # Set height as: (1) absolute height [inches], (2) from aspect str/value, (3) scaled default.
  height = h
  if aspect: height = height or (RATIOS[aspect] if isinstance(aspect, str) else aspect) * width
  height = height or scale * kh
  # Scale by number of cols/rows and add deltas.
  return np.array([c * width + dw, r * height + dh])


RGB = T.Tuple[float, float, float]
RGBA = T.Tuple[float, float, float, float]


def color(
  color: T.Union[int, str, RGB, RGBA],
  dh=0,
  ds=0,
  dl=0,
  da=0,
  h=None,
  s=None,
  l=None,
  a=None
) -> RGBA:
  """
  Transform a color by changing the lightness, hue, saturation, or alpha. Properly contrains the
  parameters of the transformed color within the valid ranges.

  Args:
    color: Base color to transform, as a string ('C0', 'blue', '0.5'), int (0 is 'C0'), or tuple of RGB or RGBA values (between 0-1).
    dh: Additive change in hue (between 0 - 360).
    ds: Additive change in saturation (between 0 - 100).
    dl: Additive change in lightness (between 0 - 100),
    da: Additive change in alpha (between 0 - 100).
    h: Fixed value in hue (between 0 - 360).
    s: Fixed value in saturation (between 0 - 100).
    l: Fixed value in lightness (between 0 - 100).
    a: Fixed value in alpha (between 0 - 100).
  """
  if isinstance(color, int):
    color = f'C{color}'
  r0, g0, b0, a0 = mcolors.ColorConverter.to_rgba(color)  # type: ignore
  h0, s0, l0 = hsluv.rgb_to_hsluv((r0, g0, b0))
  r1, g1, b1 = hsluv.hsluv_to_rgb((
      np.clip(h if h is not None else np.mod(dh + h0, 360), 0, 360),
      np.clip(s if s is not None else ds + s0, 0, 100),
      np.clip(l if l is not None else dl + l0, 0, 100),
  ))
  a1 = np.clip(a / 100 if a is not None else da / 100 + a0, 0, 1)
  return (r1, g1, b1, a1)


def pt_to_data_units(ax, x, y):
  """Convert (x, y) from point units to data units on axes `ax`."""
  t = ax.transData.inverted()
  return t.transform((x, y)) - t.transform((0, 0))


def pt_to_axes_units(ax, x, y):
  """Convert (x, y) from point units to axes units on axes `ax`."""
  t = ax.transAxes.inverted()
  return t.transform((x, y)) - t.transform((0, 0))


def data_to_axes_units(ax, x, y):
  """Convert (x, y) from data units to axes units on axes `ax`."""
  t = ax.transData + ax.transAxes.inverted()
  return t.transform((x, y))


V = T.TypeVar('V')
ValueAndKwargs = T.Union[V, T.Tuple[V, T.Dict[str, T.Any]]]


def _value_and_kwargs(x: ValueAndKwargs):
  if isinstance(x, (tuple, list)):
    assert len(x) == 2
    assert isinstance(x[-1], collections.abc.Mapping)
    return x[0], x[-1]
  else:
    return x, {}


Number = T.Union[int, float, np.number]

LOCATORS = dict(
  null=mticker.NullLocator,
  multiple=mticker.MultipleLocator,
  maxn=mticker.MaxNLocator,
)

FORMATTERS = dict(
  null=mticker.NullFormatter,
  scalar=mticker.ScalarFormatter,
  eng=mticker.EngFormatter,
  percent=mticker.PercentFormatter,
)

LocatorSpec = T.Union[
  ValueAndKwargs[str],
  T.Union[int, float],
  mticker.Locator,
]

FormatterSpec = T.Union[
  ValueAndKwargs[str],
  int,
  T.Callable[[Number, Number], str],
  mticker.Formatter,
]

TicksSpec = T.Union[
  T.Sequence[Number] | np.ndarray,
  tuple[T.Sequence[Number], T.Sequence[str]],
  tuple[T.Sequence[Number], T.Sequence[str], dict[str, T.Any]],
]

LimitsSpec = T.Union[
  tuple[Number | None, Number | None],
  tuple[Number, Number, Number],
]


def _parse_locator(locator: LocatorSpec, name: str = '') -> mticker.Locator:
  if isinstance(locator, (str, tuple, list)):
    v, kws = _value_and_kwargs(locator)
    return LOCATORS[v](**kws)
  elif isinstance(locator, (int, float)):
    return mticker.MultipleLocator(locator)
  elif isinstance(locator, mticker.Locator):
    return locator
  else:
    raise ValueError(f'Invalid {name}: {locator}')


def _parse_formatter(formatter: FormatterSpec,
                     name: str = '') -> T.Union[str, T.Callable, mticker.Formatter]:
  if isinstance(formatter, (str, tuple, list)):
    v, kws = _value_and_kwargs(formatter)
    return FORMATTERS[v](**kws) if v in FORMATTERS else v
  elif isinstance(formatter, int):
    f = mticker.ScalarFormatter()
    f.set_powerlimits((formatter, formatter))
    return f
  elif isinstance(formatter, mticker.Formatter) or callable(formatter):
    return formatter
  else:
    raise ValueError(f'Invalid {name}: {formatter}')


def _parse_ticks(ticks: TicksSpec, name: str = '') -> tuple[list, dict[str, T.Any]]:
  if len(ticks) == 0 or isinstance(ticks[0], (int, float, np.number)):
    return list(ticks), dict()
  elif len(ticks) == 2:
    vals, labels = ticks
    return vals, dict(labels=labels)  # type: ignore
  elif len(ticks) == 3:
    vals, labels, kws = ticks
    return vals, dict(labels=labels, **kws)  # type: ignore
  else:
    raise ValueError(f'Invalid {name}: {ticks}')


def set(
  ax: maxes.Axes,
  *,
  xscale: ValueAndKwargs[str] | None = None,
  yscale: ValueAndKwargs[str] | None = None,
  xlocator: LocatorSpec | None = None,
  ylocator: LocatorSpec | None = None,
  xformatter: FormatterSpec | None = None,
  yformatter: FormatterSpec | None = None,
  xticks: TicksSpec | None = None,
  yticks: TicksSpec | None = None,
  xminorlocator: LocatorSpec | None = None,
  yminorlocator: LocatorSpec | None = None,
  xminorformatter: FormatterSpec | None = None,
  yminorformatter: FormatterSpec | None = None,
  xminorticks: TicksSpec | None = None,
  yminorticks: TicksSpec | None = None,
  xlim: LimitsSpec | None = None,
  ylim: LimitsSpec | None = None,
  xlabel: ValueAndKwargs[str] | None = None,
  ylabel: ValueAndKwargs[str] | None = None,
  title: ValueAndKwargs[str] | None = None,
  margins: float | tuple[Number | None, Number | None] | None = None,
  spines: str | None = None,
  aspect: T.Literal['auto', 'equal'] | float | None = None,
  legend: ValueAndKwargs[bool] | None = None,
):
  """Set axes properties in a single call."""
  # Axes margins.
  if margins is not None:
    if isinstance(margins, (int, float)):
      ax.margins(x=margins, y=margins)
    else:
      xmargin, ymargin = margins
      if xmargin is not None and ymargin is not None:
        ax.margins(x=float(xmargin), y=float(ymargin))
      elif xmargin is not None:
        ax.margins(x=float(xmargin))
      elif ymargin is not None:
        ax.margins(y=float(ymargin))

  # Axes scale (e.g. 'linear', 'log').
  if xscale is not None:
    v, kws = _value_and_kwargs(xscale)
    ax.set_xscale(v, **kws)
  if yscale is not None:
    v, kws = _value_and_kwargs(yscale)
    ax.set_yscale(v, **kws)

  # Axes tick locator.
  if xlocator is not None:
    ax.xaxis.set_major_locator(_parse_locator(xlocator, 'xlocator'))
  if ylocator is not None:
    ax.yaxis.set_major_locator(_parse_locator(ylocator, 'ylocator'))
  if xminorlocator is not None:
    ax.xaxis.set_minor_locator(_parse_locator(xminorlocator, 'xminorlocator'))
  if yminorlocator is not None:
    ax.yaxis.set_minor_locator(_parse_locator(yminorlocator, 'yminorlocator'))

  # Axes tick formatter.
  if xformatter is not None:
    ax.xaxis.set_major_formatter(_parse_formatter(xformatter, 'xformatter'))
  if yformatter is not None:
    ax.yaxis.set_major_formatter(_parse_formatter(yformatter, 'yformatter'))
  if xminorformatter is not None:
    ax.xaxis.set_minor_formatter(_parse_formatter(xminorformatter, 'xminorformatter'))
  if yminorformatter is not None:
    ax.yaxis.set_minor_formatter(_parse_formatter(yminorformatter, 'yminorformatter'))

  # Axes ticks override. (Should set after locator/formatter.)
  if xticks is not None:
    ticks, kws = _parse_ticks(xticks, 'xticks')
    ax.set_xticks(ticks, **kws)
  if yticks is not None:
    ticks, kws = _parse_ticks(yticks, 'yticks')
    ax.set_yticks(ticks, **kws)
  if xminorticks is not None:
    ticks, kws = _parse_ticks(xminorticks, 'xminorticks')
    ax.set_xticks(ticks, minor=True, **kws)
  if yminorticks is not None:
    ticks, kws = _parse_ticks(yminorticks, 'yminorticks')
    ax.set_yticks(ticks, minor=True, **kws)

  # Axes limits. (Should set after ticks.)
  if xlim is not None:
    if len(xlim) == 2:
      x0, x1 = xlim
      ax.set_xlim(x0, x1)  # type: ignore
    elif len(xlim) == 3:
      x0, x1, dx = xlim
      ax.set_xticks(np.arange(x0, x1 + dx, step=dx))
      ax.set_xlim(x0, x1)  # type: ignore
    else:
      raise ValueError(f'Invalid xlim: {xlim}')
  if ylim is not None:
    if len(ylim) == 2:
      y0, y1 = ylim
      ax.set_ylim(y0, y1)  # type: ignore
    elif len(ylim) == 3:
      y0, y1, dy = ylim
      ax.set_yticks(np.arange(y0, y1 + dy, step=dy))
      ax.set_ylim(y0, y1)  # type: ignore
    else:
      raise ValueError(f'Invalid ylim: {ylim}')

  # Axes title and labels.
  if title is not None:
    v, kws = _value_and_kwargs(title)
    ax.set_title(v, **kws)
  if xlabel is not None:
    v, kws = _value_and_kwargs(xlabel)
    ax.set_xlabel(v, **kws)
  if ylabel is not None:
    v, kws = _value_and_kwargs(ylabel)
    ax.set_ylabel(v, **kws)

  # Axes spines.
  if spines is not None:
    for symbol, location in (('t', 'top'), ('b', 'bottom'), ('l', 'left'), ('r', 'right')):
      ax.spines[location].set_visible(symbol in spines)

  # Axes aspect.
  if aspect is not None:
    ax.set_aspect(aspect)

  # Axes legend.
  if legend is not None:
    v, kws = _value_and_kwargs(legend)
    if v:
      ax.legend(**kws)
    else:
      l = ax.get_legend()
      if l is not None: l.remove()

  return ax


@dataclasses.dataclass
class AxisInfo:
  min: float
  max: float
  center: float
  inner: float
  outer: float
  limits: tuple[float, float]
  inner_limits: tuple[float, float]
  outer_limits: tuple[float, float]
  centered_limits: tuple[float, float]
  sparse_ticks: list[float]
  dense_ticks: list[float]


def axisinfo(
  values: list[float] | np.ndarray,
  *,
  extra: list[float] = [],
  extra_within: list[float] = [],
  center: float = 0.,
) -> AxisInfo:
  """Compute axis information based on the given values and extra ticks. This is useful for setting
  axis limits, ticks, and colorbar ranges."""
  vmin = min(values)
  vmax = max(values)
  inner = min(abs(vmin - center), abs(vmax - center))
  outer = max(abs(vmin - center), abs(vmax - center))
  extra_ticks = [v for v in extra_within if vmin < v < vmax] + extra
  return AxisInfo(
    min=vmin,
    max=vmax,
    center=center,
    inner=inner,
    outer=outer,
    limits=(vmin, vmax),
    inner_limits=(center - inner, center + inner),
    outer_limits=(center - outer, center + outer),
    centered_limits=(vmin if vmin < center else center, vmax if vmax > center else center),
    sparse_ticks=list(np.unique([vmin, vmax] + extra_ticks)),  # Sorted by `np.unique`.
    dense_ticks=list(np.unique(list(values) + extra_ticks)),  # Sorted by `np.unique`.
  )
