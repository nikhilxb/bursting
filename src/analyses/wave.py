import dataclasses
import typing as T

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
from scipy import signal

import utils.plot as utp


def find_peaks(
  df: pd.DataFrame,
  prominence: float = 0.,
  t: str = 't',
  y: str = 'y_0',
) -> pd.DataFrame:
  """Find all (t, y) points corresponding to peaks in the given timeseries."""
  # Append min value to catch tail peaks, otherwise ignored because of prominence formula.
  ys = df[y]
  idxs, _ = signal.find_peaks(ys, prominence=prominence)
  return df[[t, y]].iloc[idxs]


def find_crossings(
  df: pd.DataFrame,
  t: str = 't',
  y: str = 'y_0',
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Find all (t, y) points corresponding to crossings of zero in the given timeseries.
  Crossing can be either rising (upward slope) or falling (downward slope)."""
  ys = df[y]
  idxs_rises = np.where((np.diff(np.sign(ys), append=0) == 1) |
                        (np.diff(np.sign(ys), prepend=0) == 2))[0]
  idxs_falls = np.where((np.diff(np.sign(ys), prepend=0) == -1) |
                        (np.diff(np.sign(ys), prepend=0) == -2))[0]
  return df[[t, y]].iloc[idxs_rises], df[[t, y]].iloc[idxs_falls]


def find_intervals(
  df: pd.DataFrame,
  t: str = 't',
  y: str = 'y_0',
) -> pd.DataFrame:
  """Find all (t0, t1, y0, y1) points corresponding to above/below zero intervals in the given
  timeseries."""
  ys = df[y]
  # Uses different prepend logic to include active point, not the preceding zero crossing point.
  idxs_rises = np.where((np.diff(np.sign(ys), prepend=0) == 1) |
                        (np.diff(np.sign(ys), prepend=0) == 2))[0]
  idxs_falls = np.where((np.diff(np.sign(ys), prepend=0) == -1) |
                        (np.diff(np.sign(ys), prepend=0) == -2))[0]
  rises = df[[t, y]].iloc[idxs_rises]
  falls = df[[t, y]].iloc[idxs_falls]
  # Prepend interval start if first crossing is a fall.
  try:
    if ((len(rises) == 0 and len(falls) > 0) or
        (len(rises) > 0 and len(falls) > 0 and rises.iloc[0][t] > falls.iloc[0][t])):
      rises = pd.concat([pd.DataFrame({t: [df[t].iloc[0]], y: [0]}), rises])
  except IndexError as e:
    raise IndexError(f'len(rises) == {len(rises)}, len(falls) == {len(falls)}') from e
  # Append interval end if last crossing is a rise.
  if ((len(falls) == 0 and len(rises) > 0) or
      (len(falls) > 0 and len(rises) > 0 and falls.iloc[-1][t] < rises.iloc[-1][t])):
    falls = pd.concat([falls, pd.DataFrame({t: [df[t].iloc[-1]], y: [0]})])
  # Rename columns to avoid conflicts.
  rises.columns = [f'{t}_start', f'{y}_start']
  falls.columns = [f'{t}_end', f'{y}_end']
  return pd.concat([rises.reset_index(drop=True), falls.reset_index(drop=True)], axis=1)


def find_phases(
  df: pd.DataFrame,
  period: float,
  start: float = 0.,
  count: int = 100,
  t: str = 't',
  y: str = 'y_0',
) -> tuple[pd.DataFrame, np.ndarray]:
  """Find phase of waveform peaks relative to a periodic stimulus with the given period and start
  time. If multiple peaks exist between two stimulus pulses, then the one with maximum height is
  used."""
  peaks_all = find_peaks(df, t=t, y=y)
  t_max = df[t].iloc[-1]
  peaks = []
  phases = []
  for i in range(count):
    t_start = start + period * i
    t_end = start + period * (i + 1)
    if t_end > t_max: break
    peaks_i = peaks_all.query(f'{t_start} <= {t} < {t_end}')
    if len(peaks_i) == 0:
      peaks.append((np.nan, np.nan))
      phases.append(np.nan)
    else:
      peak_max = peaks_i.iloc[peaks_i[y].argmax()]
      peaks.append((peak_max[t], peak_max[y]))
      phases.append((peak_max[t] - t_start) / period)
  return pd.DataFrame(peaks, columns=[t, y]), np.array(phases)


@dataclasses.dataclass
class WaveStats:
  T_osc: float  # Oscillation period.
  T_active: float  # Active period.
  T_quiet: float  # Quiet period.
  p_active: float  # Active phase fraction.
  p_quiet: float  # Quiet phase fraction.
  t_bounds: list[float]  # Period boundary times (active start, quiet start, active start).


@dataclasses.dataclass
class WaveInfo:
  crossings: tuple[pd.DataFrame, pd.DataFrame] | None = None
  peaks: pd.DataFrame | None = None
  stats: WaveStats | None = None


def measure_waveform(
  df: pd.DataFrame,
  t: str = 't',
  y: str = 'y_0',
  crossings: bool = True,
  peaks: bool = True,
  stats: bool = True,
  ref: int = 0,
) -> WaveInfo:
  crossings_data = find_crossings(df, t=t, y=y) if crossings else None
  peaks_data = find_peaks(df, t=t, y=y) if peaks else None
  stats_data = None
  if stats and crossings_data:
    rises, falls = crossings_data
    if (len(rises) >= 2 + ref and len(falls) >= 1 + ref  # Contains enough oscillation periods.
        and rises.iloc[0][t] < falls.iloc[0][t]  # First crossing is a rise.
       ):
      t_rise0, t_rise1 = rises.iloc[ref:ref + 2][t]
      t_fall0 = falls.iloc[ref][t]
      T_osc = t_rise1 - t_rise0
      T_active = t_fall0 - t_rise0
      T_quiet = t_rise1 - t_fall0
      stats_data = WaveStats(
        T_osc=T_osc,
        T_active=T_active,
        T_quiet=T_quiet,
        p_active=T_active / T_osc,
        p_quiet=T_quiet / T_osc,
        t_bounds=[t_rise0, t_fall0, t_rise1],
      )
  return WaveInfo(
    crossings=crossings_data,
    peaks=peaks_data,
    stats=stats_data,
  )


def plot_timeseries(
  df: pd.DataFrame,
  ax: matplotlib.axes.Axes | None = None,
  t: str = 't',
  y: str = 'y_0',
  info: WaveInfo | None = None,
  rises: bool = False,
  falls: bool = False,
  labels: bool = False,
  labels_kws: dict[str, T.Any] = {},
  peaks: pd.DataFrame | None = None,
  tstimuli: list[tuple[float, float]] | None = None,
  cstimuli: list[str | tuple[float, ...]] | str | tuple[float, ...] | None = None,
  trefs: list[float] | None = None,
  yrefs: list[float] | None = None,
  tlim: tuple[int | None, int | None] = (None, None),
  ylim: tuple[int | None, int | None] = (0, 1),
  **kwargs,
):
  if ax is None:
    _, ax = plt.subplots(figsize=utp.figsize(1, 1 / 3))

  if tstimuli is not None:
    if cstimuli is None:
      cstimuli = ['x2'] * len(tstimuli)  # type: ignore
    elif not isinstance(cstimuli, list):
      cstimuli = [cstimuli] * len(tstimuli)
    assert cstimuli is not None
    for i, (tstart, tend) in enumerate(tstimuli):
      ax.axvspan(tstart, tend, color=cstimuli[i], lw=0)

  if tlim[0] is not None:
    df = df.query(f'{tlim[0]} <= {t}')
  if tlim[1] is not None:
    df = df.query(f'{t} <= {tlim[1]}')
  ax.plot(df[t], df[y], **{**dict(lw=1.5, c='k', zorder=10), **kwargs})
  utp.set(ax, ylim=ylim)

  if info:
    if info.crossings and rises:
      ax.plot(
        info.crossings[0][t],
        info.crossings[0][y],
        marker=10,
        markersize=4,
        markeredgewidth=0,
        lw=0,
        c='k',
        zorder=15,
        clip_on=False,
        in_layout=False
      )
    if info.crossings and falls:
      ax.plot(
        info.crossings[1][t],
        info.crossings[1][y],
        marker=11,
        markersize=4,
        markeredgewidth=0,
        lw=0,
        c='k',
        zorder=15,
        clip_on=False,
        in_layout=False
      )

    # Specify x-coord in data coordinates, y-coord in axes coordinates.
    TRANSFORM = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ARROWPROPS = dict(arrowstyle='|-|', shrinkA=0, shrinkB=0, mutation_scale=1.5)
    y1 = labels_kws.get('y1', 1)  # axes coords
    y2 = labels_kws.get('y2', 1.2)  # axes coords
    size = labels_kws.get('size', 5.5)  # pts
    vpad = labels_kws.get('vpad', 3)  # pts
    hpad = labels_kws.get('hpad', 2)  # pts
    label_active = labels_kws.get('active', True)
    label_quiet = labels_kws.get('quiet', True)
    label_duty = labels_kws.get('duty', True)
    label_period = labels_kws.get('period', True)
    if labels and info.stats:
      # Calculate: (1) oscillation period, (2) active period time T and duty %, (3) quiet period time T and duty %.
      stats = info.stats

      # Horizontal arrows labeling periods.
      t0, t1, t2 = stats.t_bounds
      if label_active:
        ax.annotate(
          f'{stats.T_active:g} ({stats.p_active:.0%})' if label_duty else f'{stats.T_active:g}',
          xy=(t1, y1),
          xytext=(-hpad, vpad),
          textcoords='offset points',
          ha='right',
          xycoords=TRANSFORM,
          fontsize=size,
        )
        ax.annotate('', xy=(t0, y1), xytext=(t1, y1), arrowprops=ARROWPROPS, xycoords=TRANSFORM)

      if label_quiet:
        ax.annotate(
          f'({stats.p_quiet:.0%}) {stats.T_quiet:g}' if label_duty else f'{stats.T_quiet:g}',
          xy=(t1, y1),
          xytext=(hpad, vpad),
          textcoords='offset points',
          ha='left',
          xycoords=TRANSFORM,
          fontsize=size,
        )
        ax.annotate('', xy=(t1, y1), xytext=(t2, y1), arrowprops=ARROWPROPS, xycoords=TRANSFORM)

      if label_period:
        ax.annotate(
          f'{stats.T_osc:g}',
          xy=((t0 + t2) / 2, y2),
          xytext=(0, vpad),
          textcoords='offset points',
          ha='center',
          xycoords=TRANSFORM,
          fontsize=size,
        )
        ax.annotate('', xy=(t0, y2), xytext=(t2, y2), arrowprops=ARROWPROPS, xycoords=TRANSFORM)

      # Vertical lines separating periods.
      VLINE_KWS: dict[str, T.Any] = dict(lw=0.5, c='k', zorder=1)
      if label_period or label_active:
        ax.axvline(t0, ymax=y1, **VLINE_KWS)
      if label_active or label_quiet:
        ax.axvline(t1, ymax=y1, **VLINE_KWS)
      if label_period or label_quiet:
        ax.axvline(t2, ymax=y1, **VLINE_KWS)

    elif labels and not info.stats:
      # Spacing to ensure labeled and unlabeled axes are same size.
      ax.annotate(
        ' ',
        xy=(min(df[t]), y2),
        xytext=(0, vpad),
        textcoords='offset points',
        ha='center',
        xycoords=TRANSFORM,
        fontsize=size,
      )

  if peaks is not None:
    ax.plot(peaks[t], peaks[y], marker='.', lw=0, c='k', zorder=15, clip_on=False)

  if trefs is not None:
    for tref in trefs:
      ax.axvline(tref, lw=0.5, c='k')

  if yrefs is not None:
    for yref in yrefs:
      ax.axhline(yref, lw=0.5, c='k')
