import dataclasses
import logging
import os
import pathlib
import typing as T

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

import utils.numeric as utn
import utils.plot as utp

from .. import stimuli
from . import constant, wave

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PeriodicVariantTrial:
  # Start phase of stimulus.
  p_start: float

  # Timespans [[t_start, t_end]] for stimuli.
  stimulus: list[tuple[float, float]]

  # Waveform timeseries [(t, y)].
  waveform: pd.DataFrame

  # Waveform peaks [(t, y)].
  peaks: pd.DataFrame

  # Waveform phases [p] with `pulse_count` values. Phases are relative
  # to the periodic stimulus and within range [0, 1].
  phases: np.ndarray


@dataclasses.dataclass
class PeriodicVariant:
  # Pulse width of stimulus.
  width: float

  # Strength of stimulus.
  strength: float

  # Pulse period of stimulus.
  period: float

  # Phase order, averaged over `pulse_count` pulses and phase inits.
  phase_order: complex

  # Phase coherence, magnitude of `p_order`.
  phase_coherence: float

  # Trials for every init phase.
  trials: list[PeriodicVariantTrial]


@dataclasses.dataclass
class PeriodicData:
  # Column key for time `t`.
  t: str

  # Column key for signal output `y`.
  y: str

  # Pulse widths of all stimulus variants.
  widths: list[float]

  # Pulse strengths of all stimulus variants.
  strengths: list[float]

  # Pulse periods of all stimulus variants.
  periods: list[float]

  # Pulse count across all stimulus variants.
  pulse_count: int

  # Pulse inital phase spacing across all stimulus variants.
  init_every: float

  # Waveform timeseries [(t, y)].
  df_baseline: pd.DataFrame

  # Waveform information.
  wave_baseline: wave.WaveInfo

  # Measurements for each [(width, strength, period)] run.
  variants: dict[tuple[float, float, float], PeriodicVariant]


def measure(
  simulate: T.Callable[..., pd.DataFrame],
  t: str = 't',
  y: str = 'y_0',
  widths: T.Sequence[float] = (100.,),
  strengths_range: tuple[float, float, float] = (0., 1., 0.1),
  periods_range: tuple[float, float, float] = (0.75, 1.25, 0.05),
  pulse_count: int = 10,
  init_every: float = 0.1,
) -> PeriodicData:
  # Measure baseline waveform period/boundaries, if possible.
  df_baseline = simulate(stimulus=stimuli.constant(0.))
  wave_baseline = wave.measure_waveform(df_baseline, t=t, y=y)

  # Measure periodic-stimulated waveforms.
  widths = sorted(widths)  # widths[0] < widths[1] < ...
  strengths = list(utn.floatrange(*strengths_range))
  periods = list(utn.floatrange(*periods_range))
  if wave_baseline.stats is None:
    log.warn(f'Baseline waveform stats not measureable. No variants will be run.')
  if len(widths) == 0 or len(strengths) == 0 or len(periods) == 0 or wave_baseline.stats is None:
    return PeriodicData(
      t=t,
      y=y,
      widths=widths,
      strengths=strengths,
      periods=periods,
      pulse_count=pulse_count,
      init_every=init_every,
      df_baseline=df_baseline,
      wave_baseline=wave_baseline,
      variants={},
    )

  wave_start = wave_baseline.stats.t_bounds[0]
  wave_end = wave_baseline.stats.t_bounds[2]
  wave_period = wave_baseline.stats.T_osc

  # Calculate minimum simulation time needed for tests, bounded using max phase and max period.
  min_sim_time = wave_end + pulse_count * periods_range[1] * wave_period
  actual_sim_time = df_baseline[t].iloc[-1]
  simulate_kwargs = dict(T=np.ceil(min_sim_time)) if actual_sim_time < min_sim_time else dict()

  @ray.remote
  def measure_variant(width: float, strength: float, period: float) -> PeriodicVariant:
    phase_order = 0j  # Complex number to calculate circular mean of phases, across pulses and phase inits.
    phase_count = 0
    trials = []
    for p_start in np.arange(0, 1, init_every):
      t_start = wave_start + p_start * wave_period
      stimulus = [(
        t_start + n * period * wave_period,
        t_start + n * period * wave_period + width,
      ) for n in range(pulse_count)]
      waveform = simulate(
        stimulus=stimuli.periodic(
          start=t_start,
          period=period * wave_period,
          width=width,
          high=strength,
          low=0.,
          count=pulse_count,
        ),
        **simulate_kwargs,
      )
      peaks, phases = wave.find_phases(waveform, t=t, y=y, start=t_start, period=period * wave_period, count=pulse_count)
      for p in phases:
        if not np.isnan(p):
          phase_order += np.exp(2j * np.pi * p)
          phase_count += 1
      trials.append(
        PeriodicVariantTrial(
          p_start=p_start,
          stimulus=stimulus,
          waveform=waveform,
          peaks=peaks,
          phases=phases,
        )
      )
    if phase_count > 0:
      phase_order /= phase_count
    return PeriodicVariant(
      width=width,
      strength=strength,
      period=period,
      phase_order=phase_order,
      phase_coherence=abs(phase_order),
      trials=trials,
    )

  variants_refs = {(width, strength, period): measure_variant.remote(width, strength, period)
                   for width in widths
                   for strength in strengths
                   for period in periods}
  return PeriodicData(
    t=t,
    y=y,
    widths=widths,
    strengths=strengths,
    periods=periods,
    pulse_count=pulse_count,
    init_every=init_every,
    df_baseline=df_baseline,
    wave_baseline=wave_baseline,
    variants={
      k: ray.get(v)
      for k, v in variants_refs.items()
    },
  )


# ==================================================================================================
# Plots


def plot_stimuli_range(
  data: PeriodicData,
  timeseries_kws: dict[str, T.Any] = {},
  axs: T.Any = None,
):
  if axs is None:
    _, axs = plt.subplots(3, 1, figsize=utp.figsize(1, h=0.75, r=3), sharex=True, sharey=True, squeeze=False, layout='constrained')

  assert data.wave_baseline.stats is not None
  wave_start = data.wave_baseline.stats.t_bounds[0]
  wave_end = data.wave_baseline.stats.t_bounds[2]
  wave_period = data.wave_baseline.stats.T_osc

  for width in data.widths:
    ax = axs[0, 0]
    wave.plot_timeseries(
      data.df_baseline,
      ax=ax,
      t=data.t,
      y=data.y,
      info=data.wave_baseline,
      tstimuli=[(
        wave_start + n * data.periods[0] * wave_period,
        wave_start + n * data.periods[0] * wave_period + width,
      ) for n in range(data.pulse_count)],
      **{
        **dict(rises=True, falls=True),
        **timeseries_kws,
      },
    )
    ax.axvline(wave_start, lw=0.5, c='C2')
    ax.axvline(wave_end + data.pulse_count * data.periods[0] * wave_period, lw=0.5, c='C3')
    utp.set(ax, title=rf'$\mathtt{{period}}$ = {data.periods[0]:.2f}')

    ax = axs[1, 0]
    wave.plot_timeseries(
      data.df_baseline,
      ax=ax,
      t=data.t,
      y=data.y,
      info=data.wave_baseline,
      tstimuli=[(
        wave_start + n * 1.0 * wave_period,
        wave_start + n * 1.0 * wave_period + width,
      ) for n in range(data.pulse_count)],
      **{
        **dict(rises=True, falls=True),
        **timeseries_kws,
      },
    )
    ax.axvline(wave_start, lw=0.5, c='C2')
    ax.axvline(wave_end + data.pulse_count * 1.0 * wave_period, lw=0.5, c='C3')
    utp.set(ax, title=rf'$\mathtt{{period}}$ = {1.:.2f}')

    ax = axs[2, 0]
    wave.plot_timeseries(
      data.df_baseline,
      ax=ax,
      t=data.t,
      y=data.y,
      info=data.wave_baseline,
      tstimuli=[(
        wave_start + n * data.periods[-1] * wave_period,
        wave_start + n * data.periods[-1] * wave_period + width,
      ) for n in range(data.pulse_count)],
      **{
        **dict(rises=True, falls=True),
        **timeseries_kws,
      },
    )
    ax.axvline(wave_start, lw=0.5, c='C2')
    ax.axvline(wave_end + data.pulse_count * data.periods[-1] * wave_period, lw=0.5, c='C3')
    utp.set(ax, title=rf'$\mathtt{{period}}$ = {data.periods[-1]:.2f}')


def plot_waveforms(
  data: PeriodicData,
  width: float,
  strength: float,
  period: float,
  strengths: list[float] | None = None,
  trials: list[int] = [],
  timeseries_kws: dict[str, T.Any] = {},
  phase_kws: dict[str, T.Any] = {},
  phases: bool = True,
  cmap: str | mpl.colors.Colormap = 'coolwarm',
  axs: T.Any = None,
):
  strengths = strengths or data.strengths
  axis_strengths = utp.axisinfo(strengths, extra_within=[0.], center=0.)
  color_strengths = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap,
    norm=mpl.colors.Normalize(*axis_strengths.outer_limits),
  )
  color = color_strengths.to_rgba(strength)

  variant = data.variants[(width, strength, period)]
  trial_idxs = trials or list(range(len(variant.trials)))
  num_plots = len(trial_idxs)

  if axs is None:
    _, axs = plt.subplots(num_plots, 2, figsize=utp.figsize(1, h=0.75, r=num_plots), width_ratios=[21, 3], squeeze=False, layout='constrained')
  elif callable(axs):
    axs = axs(num_plots)

  for idx, trial_idx in enumerate(trial_idxs):
    trial = variant.trials[trial_idx]
    ax = axs[idx, 0]
    wave.plot_timeseries(
      trial.waveform,
      ax=ax,
      t=data.t,
      y=data.y,
      peaks=trial.peaks,
      tstimuli=trial.stimulus,
      cstimuli='r2' if strength > 0 else 'b2' if strength < 0 else 'x2',
      c=color,
      **timeseries_kws,
    )

    if phases:
      ax = axs[idx, 1]
      ax.plot(
        np.arange(1, len(trial.phases) + 1), trial.phases, '.-', **{
          **dict(c='k'), **phase_kws
        }
      )
      utp.set(ax, ylim=(0, 1))


def plot_arnold_tongue(
  data: PeriodicData,
  widths: list[float] | None = None,
  strengths: list[float] | None = None,
  periods: list[float] | None = None,
  slice_every: tuple[float | None, float | None] = (None, None),
  colorbar_phases: bool = True,
  colorbar_strengths: bool = True,
  colorbar_periods: bool = True,
  colorbar_phases_kws: dict[str, T.Any] = {},
  colorbar_strengths_kws: dict[str, T.Any] = {},
  colorbar_periods_kws: dict[str, T.Any] = {},
  cmap_phases: str | mpl.colors.Colormap = 'mako',
  cmap_strengths: str | mpl.colors.Colormap = 'coolwarm',
  cmap_periods: str | mpl.colors.Colormap = 'icefire',
  axs: T.Any = None,
):
  widths = widths or data.widths
  strengths = strengths or data.strengths
  periods = periods or data.periods
  slice_strength = slice_every[0] != 0
  slice_period = slice_every[1] != 0
  slices_strengths = utn.select_every(strengths, slice_every[0] or 0)
  slices_periods = utn.select_every(periods, slice_every[1] or 0)
  axis_strengths = utp.axisinfo(strengths, extra_within=[0.], center=0.)
  axis_periods = utp.axisinfo(periods, extra_within=[1.], center=1.)
  color_phases = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap_phases,
    norm=mpl.colors.Normalize(0, 1),
  )
  color_strengths = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap_strengths,
    norm=mpl.colors.Normalize(*axis_strengths.outer_limits),
  )
  color_periods = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap_periods,
    norm=mpl.colors.Normalize(*axis_periods.outer_limits),
  )

  if axs is None:
    num_rows = 1 + slice_strength + slice_period
    _, axs = plt.subplots(num_rows, len(widths), figsize=utp.figsize(w=2.25, h=2, r=3, c=len(widths)), sharex='row', sharey='row', squeeze=False)

  for w, width in enumerate(widths):
    # Phase coherence heatmap.
    ax = axs[0, w]
    ax1 = ax
    grid = np.zeros((len(strengths), len(periods)))
    for s, strength in enumerate(strengths):
      for p, period in enumerate(periods):
        grid[s, p] = data.variants[(width, strength, period)].phase_coherence
    cnt = ax.contourf(
      periods, strengths, grid, levels=20, cmap=color_phases.cmap, norm=color_phases.norm
    )
    for c in cnt.collections:
      c.set_edgecolor('face')  # Prevents aliasing with pdf renderer.
    utp.set(
      ax,
      title=f'Stimulus width = {width:3g}',
      xlabel='Stimulus period',
      ylabel='Stimulus strength' if w == 0 else None,
      xticks=axis_periods.sparse_ticks,
      yticks=axis_strengths.sparse_ticks,
      xminorticks=slices_periods,
      yminorticks=slices_strengths,
      spines='tblr',
    )

    # Horizontal slices (strength).
    if slice_strength:
      ax = axs[1, w]
      ax2 = ax
      for strength in slices_strengths:
        profile = [data.variants[(width, strength, period)].phase_coherence for period in periods]
        ax.plot(periods, profile, c=color_strengths.to_rgba(strength))
      utp.set(
        ax,
        xlabel='Stimulus period',
        ylabel='Phase coherence' if w == 0 else None,
        ylim=(0, 1, 0.5),
        yminorlocator=0.1,
        xticks=axis_periods.sparse_ticks,
        xminorticks=slices_periods,
      )

    # Vertical slices (period).
    if slice_period:
      ax = axs[1 + slice_strength, w]
      ax3 = ax
      for period in slices_periods:
        profile = [
          data.variants[(width, strength, period)].phase_coherence for strength in strengths
        ]
        ax.plot(strengths, profile, c=color_periods.to_rgba(period))
      utp.set(
        ax,
        xlabel='Stimulus strength',
        ylabel='Phase coherence' if w == 0 else None,
        ylim=(0, 1, 0.5),
        yminorlocator=0.1,
        xticks=axis_strengths.sparse_ticks,
        xminorticks=slices_strengths,
      )

  if colorbar_phases:
    if hasattr(ax1, 'colorbar'):
      label = colorbar_phases_kws.pop('label', 'Phase coherence')
      label_kws = colorbar_phases_kws.pop('label_kws', dict(rotation=270, va='bottom'))
      cb1 = ax1.colorbar(
        cnt, **{
          **dict(ticks=(0, 0.5, 1), minorticks=0.1),
          **colorbar_phases_kws,
        }
      )
      cb1.set_label(label, **label_kws)
    else:
      cb1 = plt.colorbar(
        color_phases,
        ax=axs[0, :].ravel().tolist(),
        **{
          **dict(label='Phase coherence'), **colorbar_phases_kws
        },
      )
      utp.set(cb1.ax, ylim=(0, 1, 0.5), yminorlocator=0.1)

  if slice_strength and colorbar_strengths:
    if hasattr(ax2, 'colorbar'):
      label = colorbar_strengths_kws.pop('label', 'Stimulus strength')
      label_kws = colorbar_strengths_kws.pop('label_kws', dict(rotation=270, va='bottom'))
      cb2 = ax2.colorbar(
        color_strengths,
        **{
          **dict(ticks=axis_strengths.sparse_ticks, minorticks=slices_strengths),
          **colorbar_strengths_kws,
        }
      )
      cb2.set_label(label, **label_kws)
    else:
      cb2 = plt.colorbar(
        color_strengths,
        ax=axs[1, :].ravel().tolist(),
        **{
          **dict(label='Stimulus strength'), **colorbar_strengths_kws
        },
      )
      utp.set(
        cb2.ax,
        yticks=axis_strengths.sparse_ticks,
        yminorticks=slices_strengths,
        ylim=axis_strengths.centered_limits,
      )

  if slice_period and colorbar_periods:
    if hasattr(ax3, 'colorbar'):
      label = colorbar_periods_kws.pop('label', 'Stimulus period')
      label_kws = colorbar_periods_kws.pop('label_kws', dict(rotation=270, va='bottom'))
      cb3 = ax3.colorbar(
        color_periods,
        **{
          **dict(ticks=axis_periods.sparse_ticks, minorticks=slices_periods),
          **colorbar_periods_kws,
        }
      )
      cb3.set_label(label, **label_kws)
    else:
      cb3 = plt.colorbar(
        color_periods,
        ax=axs[1 + slice_strength, :].ravel().tolist(),
        **{
          **dict(label='Stimulus period'), **colorbar_periods_kws
        },
      )
      utp.set(
        cb3.ax,
        yticks=axis_periods.sparse_ticks,
        yminorticks=slices_periods,
        ylim=axis_periods.centered_limits,
      )


def plot(
  data: PeriodicData,
  waveform_every: tuple[float | None, float | None] = (None, None),
  slice_every: tuple[float | None, float | None] = (None, None),
  x_nullcline: T.Callable | None = None,
  y_nullcline: T.Callable | None = None,
  timeseries_kws: dict[str, T.Any] = {},
  phase_kws: dict[str, T.Any] = {},
  output_dir: os.PathLike | str = '.',
) -> None:
  output_dir = pathlib.Path(output_dir)
  os.makedirs(output_dir, exist_ok=True)

  utp.setup_plotting()

  # Plot baseline waveform (timeseries, phase portrait).
  constant.plot_baseline(
    T.cast(constant.ConstantData, data),
    x_nullcline=x_nullcline,
    y_nullcline=y_nullcline,
    timeseries_kws=timeseries_kws,
    phase_kws=phase_kws,
  )
  plt.savefig(output_dir / 'baseline.pdf')
  plt.close()

  if len(data.widths) == 0 or len(data.strengths) == 0 or len(
      data.periods) == 0 or data.wave_baseline.stats is None:
    return

  # Plot stimuli range on baseline waveform (timeseries).
  plot_stimuli_range(
    data,
    timeseries_kws=timeseries_kws,
  )
  plt.savefig(output_dir / 'stimuli_range.pdf')
  plt.close()

  # Plot periodic-stimulated waveforms.
  @ray.remote
  def plot_waveforms_parallel(data: PeriodicData, width: float, strength: float, period: float):
    utp.setup_plotting()
    plot_waveforms(
      data,
      width=width,
      strength=strength,
      period=period,
      timeseries_kws=timeseries_kws,
    )
    variant = data.variants[(width, strength, period)]
    plt.suptitle(
      rf'$\mathtt{{width}}$ = {width:.2f}, $\mathtt{{strength}}$ = {strength:.2f}, $\mathtt{{period}}$ = {period:.2f} | Phase coherence = {variant.phase_coherence:4.2f}'
    )
    plt.savefig(output_dir / f'waveforms_{width=:.2f}_{strength=:.2f}_{period=:.2f}.pdf')
    plt.close()

  data_ref = ray.put(data)
  ray.get([
    plot_waveforms_parallel.remote(data_ref, width, strength, period)
    for width in data.widths
    for strength in utn.select_every(data.strengths, waveform_every[0] or 0)
    for period in utn.select_every(data.periods, waveform_every[1] or 0)
  ])

  # Plot Arnold tongue (stimulus period vs. stimulus strength vs. phase coherence), and vertical/horizontal slices (stimulus period, stimulus strength).
  plot_arnold_tongue(data, slice_every=slice_every)
  plt.savefig(output_dir / 'arnold_tongue.pdf')
