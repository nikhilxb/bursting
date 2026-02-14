import dataclasses
import functools
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

from .. import plots, stimuli
from . import constant, wave

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PulseVariantTrial:
  # Start time of stimulus.
  t_start: float

  # End time of stimulus.
  t_end: float

  # Start phase of stimulus.
  p_start: float

  # Oscillation period of stimulated waveform.
  T_stim: float | None

  # Difference in oscillation periods between stimulated and baseline waveforms.
  T_diff: float | None

  # Waveform timeseries [(t, y)].
  df_stim: pd.DataFrame

  # Waveform information.
  wave_stim: wave.WaveInfo


@dataclasses.dataclass
class PulseVariant:
  # Pulse width of stimulus.
  width: float

  # Strength of stimulus.
  strength: float

  # Trials for every pulse phase.
  trials: list[PulseVariantTrial]


@dataclasses.dataclass
class PulseData:
  # Column key for time `t`.
  t: str

  # Column key for signal output `y`.
  y: str

  # Pulse widths of all stimulus variants.
  widths: list[float]

  # Pulse strengths of all stimulus variants.
  strengths: list[float]

  # Pulse spacing across all stimulus variants.
  pulse_every: float

  # Waveform timeseries [(t, y)].
  df_baseline: pd.DataFrame

  # Waveform measurement, if measurable.
  wave_baseline: wave.WaveInfo

  # Measurements for each [(width, strength)] run.
  variants: dict[tuple[float, float], PulseVariant]


def measure(
  simulate: T.Callable[..., pd.DataFrame],
  t: str = 't',
  y: str = 'y_0',
  widths: T.Sequence[float] = (100.,),
  strengths: T.Sequence[float] = (1.,),
  pulse_every: float = 100.,
) -> PulseData:
  # Measure baseline waveform period/boundaries, if possible.
  df_baseline = simulate(stimulus=stimuli.constant(0.))
  wave_baseline = wave.measure_waveform(df_baseline, t=t, y=y, peaks=False)

  # Measure pulse-stimulated waveforms.
  widths = sorted(widths)  # widths[0] < widths[1] < ...
  strengths = sorted(strengths)  # strengths[0] < strengths[1] < ...
  if wave_baseline.stats is None:
    log.warn(f'Baseline waveform stats not measureable. No variants will be run.')
  if len(widths) == 0 or len(strengths) == 0 or wave_baseline.stats is None:
    return PulseData(
      t=t,
      y=y,
      widths=widths,
      strengths=strengths,
      pulse_every=pulse_every,
      df_baseline=df_baseline,
      wave_baseline=wave_baseline,
      variants={},
    )

  wave_start = wave_baseline.stats.t_bounds[0]
  wave_end = wave_baseline.stats.t_bounds[2]
  wave_period = wave_baseline.stats.T_osc

  @ray.remote
  def measure_variant(width: float, strength: float) -> PulseVariant:
    trials = []
    for t_start in np.arange(wave_start, wave_end, pulse_every):
      df_stim = simulate(stimulus=stimuli.pulse(start=t_start, width=width, high=strength, low=0.))
      wave_stim = wave.measure_waveform(df_stim, t=t, y=y)
      trials.append(
        PulseVariantTrial(
          t_start=t_start,
          t_end=t_start + width,
          p_start=(t_start - wave_start) / wave_period,
          T_stim=wave_stim.stats and wave_stim.stats.T_osc,
          T_diff=wave_stim.stats and (wave_stim.stats.T_osc - wave_period) / wave_period,
          df_stim=df_stim,
          wave_stim=wave_stim,
        )
      )
    return PulseVariant(
      width=width,
      strength=strength,
      trials=trials,
    )

  variants_refs = {(width, strength): measure_variant.remote(width, strength)
                   for width in widths
                   for strength in strengths}
  return PulseData(
    t=t,
    y=y,
    widths=widths,
    strengths=strengths,
    pulse_every=pulse_every,
    df_baseline=df_baseline,
    wave_baseline=wave_baseline,
    variants={
      k: ray.get(v)
      for k, v in variants_refs.items()
    },
  )


# ==================================================================================================
# Plots


def plot_waveforms(
  data: PulseData,
  width: float,
  strength: float,
  strengths: list[float] | None = None,
  phases: list[float] | None = None,
  waveform_every: float | None = None,
  x_nullcline: T.Callable | None = None,
  y_nullcline: T.Callable | None = None,
  timeseries_kws: dict[str, T.Any] = {},
  phase_kws: dict[str, T.Any] = {},
  baseline: bool = True,
  portraits: bool = True,
  cmap: str | mpl.colors.Colormap = 'coolwarm',
  axs: T.Any = None,
):
  strengths = strengths or data.strengths
  axis_strengths = utp.axisinfo(strengths, extra_within=[0.], center=0.)
  color_strengths = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap,
    norm=mpl.colors.Normalize(*axis_strengths.outer_limits),
  )

  variant = data.variants[(width, strength)]
  width, strength = variant.width, variant.strength
  color = color_strengths.to_rgba(strength)
  trial_idxs = (
    utn.select_geq([trial.p_start for trial in variant.trials], phases, idxs=True)[1] if phases else
    utn.select_every([trial.t_start for trial in variant.trials], waveform_every or 0, idxs=True)[1]
  )
  num_plots = len(trial_idxs)

  if axs is None:
    _, axs = plt.subplots(num_plots, 2, figsize=utp.figsize(1, h=0.75, r=num_plots), width_ratios=[21, 3], squeeze=False, layout='constrained')
  elif callable(axs):
    axs = axs(num_plots)

  for idx, trial_idx in enumerate(trial_idxs):
    trial = variant.trials[trial_idx]
    ax = axs[idx, 0]
    if baseline:
      wave.plot_timeseries(
        data.df_baseline,
        ax=ax,
        t=data.t,
        y=data.y,
        lw=0.5,
        c='x',
      )
    wave.plot_timeseries(
      trial.df_stim,
      ax=ax,
      t=data.t,
      y=data.y,
      info=trial.wave_stim,
      tstimuli=[(trial.t_start, trial.t_end)],
      cstimuli=['r2' if strength > 0 else 'b2' if strength < 0 else 'x2'],
      **{
        **dict(rises=True, falls=True, labels=False, lw=1.5),
        **timeseries_kws,
        **dict(c=color),
      },
    )

    if portraits:
      ax = axs[idx, 1]
      x_nullclines = None
      if x_nullcline:
        x_nullclines = [functools.partial(x_nullcline, i=strength)]
        if baseline: x_nullclines.insert(0, functools.partial(x_nullcline, i=0.))
      y_nullclines = None
      if y_nullcline:
        y_nullclines = [functools.partial(y_nullcline, i=strength)]
        if baseline: y_nullclines.insert(0, functools.partial(y_nullcline, i=0.))
      plots.plot_phase_portrait(
        trial.df_stim,
        ax=ax,
        x_nullcline=x_nullclines,
        y_nullcline=y_nullclines,
        **{
          **dict(
            delta_nullcline=100,
            trail_style='-',
            initial=False,
            final=False,
            legend=False,
            x_nullcline_kws=dict(lw=0.5, zorder=9),
            y_nullcline_kws=dict(lw=0.5, zorder=8),
          ),
          **phase_kws,
          **dict(
            trail_kws={
              **dict(lw=1.25, c=color), **phase_kws.get('trail_kws', {})
            },
          ),
        }
      )


def plot_phase_response_curve(
  data: PulseData,
  widths: list[float] | None = None,
  strengths: list[float] | None = None,
  colorbar: bool = True,
  colorbar_kws: dict[str, T.Any] = {},
  cmap: str | mpl.colors.Colormap = 'coolwarm',
  axs: T.Any = None,
):
  widths = widths or data.widths
  strengths = strengths or data.strengths
  axis_strengths = utp.axisinfo(strengths, extra_within=[0.], center=0.)
  color_strengths = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap,
    norm=mpl.colors.Normalize(*axis_strengths.outer_limits),
  )

  if axs is None:
    _, axs = plt.subplots(1, len(widths), figsize=utp.figsize(w=2.25, h=3, c=len(widths)), sharex=True, sharey=True, squeeze=False, layout='constrained')

  assert data.wave_baseline.stats is not None
  lines = []
  for w, width in enumerate(widths):
    ax = axs[0, w]
    ax.hlines(0, 0, 1, lw=0.5, color='k', zorder=3)
    ax.vlines(data.wave_baseline.stats.p_active, -1, 1, lw=0.5, ls='--', color='k', zorder=1)
    lines = []
    ordered_strengths = (
      sorted([x for x in strengths if x <= 0], reverse=True) +
      sorted([x for x in strengths if x > 0])
    )
    for s, strength in enumerate(ordered_strengths):
      p_starts, T_diffs = zip(*[(trial.p_start, trial.T_diff) for trial in data.variants[(width, strength)].trials])
      g = ax.plot(p_starts, T_diffs, '.-', lw=1, c=color_strengths.to_rgba(strength))
      lines.append((strength, g[0]))
    lines = [line for _, line in sorted(lines, key=lambda x: x[0])]
    utp.set(
      ax,
      xlim=(0, 1),
      ylim=(-1, 1),
      xlabel='Stimulus phase',
      title=rf'$\mathtt{{width}}$ = {width:g}',
      ylabel="Phase shift (T' – T) / T" if w == 0 else None,
    )
  if colorbar:
    if hasattr(ax, 'colorbar'):
      label_kws = colorbar_kws.pop('label_kws', dict(rotation=270, va='bottom'))
      cb = ax.colorbar(lines, sorted(strengths), **colorbar_kws)
      cb.set_label(colorbar_kws.get('label', 'Stimulus strength'), **label_kws)
    else:
      cb = plt.colorbar(
        color_strengths,
        ax=axs.ravel().tolist(),
        **{
          **dict(label='Stimulus strength', shrink=0.5), **colorbar_kws
        },
      )
      utp.set(
        cb.ax,
        yticks=axis_strengths.dense_ticks,
        yminorticks=data.strengths,
        ylim=axis_strengths.centered_limits,
      )


def plot(
  data: PulseData,
  waveform_every: float | None = None,
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

  if len(data.widths) == 0 or len(data.strengths) == 0 or data.wave_baseline.stats is None: return

  # Plot pulse-stimulated waveforms (timeseries).
  @ray.remote
  def plot_waveforms_parallel(data: PulseData, width: float, strength: float):
    utp.setup_plotting()
    plot_waveforms(
      data,
      width=width,
      strength=strength,
      waveform_every=waveform_every,
      x_nullcline=x_nullcline,
      y_nullcline=y_nullcline,
      timeseries_kws=timeseries_kws,
      phase_kws=phase_kws,
    )
    plt.suptitle(rf'$\mathtt{{width}}$ = {width:.2f}, $\mathtt{{strength}}$ = {strength:.2f}')
    plt.savefig(output_dir / f'waveforms_{width=:.2f}_{strength=:.2f}.pdf')
    plt.close()

  data_ref = ray.put(data)
  ray.get([
    plot_waveforms_parallel.remote(data_ref, width,
                                   strength) for width in data.widths for strength in data.strengths
  ])

  # Plot period response curves (normalized period change).
  plot_phase_response_curve(data)
  plt.savefig(output_dir / 'phase_response_curve.pdf')
  plt.close()
