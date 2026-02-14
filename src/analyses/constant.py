import dataclasses
import functools
import os
import pathlib
import typing as T

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

import utils.plot as utp

from .. import plots, stimuli
from . import wave


@dataclasses.dataclass
class ConstantVariant:
  # Strength of stimulus.
  strength: float

  # Waveform timeseries [(t, y)].
  df_stim: pd.DataFrame

  # Waveform information.
  wave_stim: wave.WaveInfo


@dataclasses.dataclass
class ConstantData:
  # Column key for time `t`.
  t: str

  # Column key for signal output `y`.
  y: str

  # Constant strengths of all stimulus variants.
  strengths: list[float]

  # Waveform timeseries [(t, y)].
  df_baseline: pd.DataFrame

  # Waveform information.
  wave_baseline: wave.WaveInfo

  # Measurements for each [strength] run.
  variants: dict[float, ConstantVariant]


def measure(
  simulate: T.Callable[..., pd.DataFrame],
  t: str = 't',
  y: str = 'y_0',
  strengths: T.Sequence[float] = (-1., 1.),
) -> ConstantData:
  # Measure baseline waveform period/boundaries, if possible.
  df_baseline = simulate(stimulus=stimuli.constant(0.))
  wave_baseline = wave.measure_waveform(df_baseline, t=t, y=y, peaks=False)

  # Measure constant-stimulated waveforms.
  strengths = sorted(set(strengths) | set([0.]))  # strengths[0] < strengths[1] < ...
  # Note: `strengths` will always contain at least the zero element, and `wave_baseline` is allowed
  # to be None, as no further calculations/plots depend on it.

  @ray.remote
  def measure_variant(strength: float) -> ConstantVariant:
    df_stim = simulate(stimulus=stimuli.constant(strength))
    wave_stim = wave.measure_waveform(df_stim, t=t, y=y)
    return ConstantVariant(
      strength=strength,
      df_stim=df_stim,
      wave_stim=wave_stim,
    )

  variants_refs = {strength: measure_variant.remote(strength) for strength in strengths}
  return ConstantData(
    t=t,
    y=y,
    strengths=strengths,
    df_baseline=df_baseline,
    wave_baseline=wave_baseline,
    variants={
      k: ray.get(v)
      for k, v in variants_refs.items()
    },
  )


# ==================================================================================================
# Plots


def plot_baseline(
  data: ConstantData,
  x_nullcline: T.Callable | None = None,
  y_nullcline: T.Callable | None = None,
  timeseries_kws: dict[str, T.Any] = {},
  phase_kws: dict[str, T.Any] = {},
  axs: T.Any = None,
) -> None:
  if axs is None:
    _, axs = plt.subplots(1, 2, figsize=utp.figsize(1, h=1), width_ratios=[21, 3], layout='constrained')
  wave.plot_timeseries(
    data.df_baseline,
    ax=axs[0],
    t=data.t,
    y=data.y,
    info=data.wave_baseline,
    **{
      **dict(rises=True, falls=True, labels=True),
      **timeseries_kws,
    },
  )
  plots.plot_phase_portrait(
    data.df_baseline,
    ax=axs[1],
    x_nullcline=x_nullcline,
    y_nullcline=y_nullcline,
    **{
      **dict(
        delta_nullcline=100,
        trail_style='-',
        trail_kws=dict(lw=1.25),
        final=False,
        legend=False,
        x_nullcline_kws=dict(lw=0.5),
        y_nullcline_kws=dict(lw=0.5),
      ),
      **phase_kws,
    },
  )


def plot_waveforms(
  data: ConstantData,
  x_nullcline: T.Callable | None = None,
  y_nullcline: T.Callable | None = None,
  timeseries_kws: dict[str, T.Any] = {},
  phase_kws: dict[str, T.Any] = {},
  strengths: list[float] | None = None,
  baseline: bool = False,
  portraits: bool = True,
  labels: bool = True,
  cmap: str = 'coolwarm',
  axs: T.Any = None,
) -> None:
  strengths = sorted(strengths or data.strengths, reverse=True)
  axis_strengths = utp.axisinfo(strengths, extra_within=[0.], center=0.)
  color_strengths = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap,
    norm=mpl.colors.Normalize(*axis_strengths.outer_limits),
  )

  if axs is None:
    _, axs = plt.subplots(
      len(strengths), 2, figsize=utp.figsize(1, h=1, r=len(strengths)), width_ratios=[21, 3], squeeze=False, layout='constrained')

  for idx, strength in enumerate(strengths):
    variant = data.variants[strength]
    color = color_strengths.to_rgba(strength) if strength != 0 else 'k'

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
      variant.df_stim,
      ax=ax,
      t=data.t,
      y=data.y,
      info=variant.wave_stim,
      **{
        **dict(rises=True, falls=True, labels=True, lw=1.5),
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
        variant.df_stim,
        ax=ax,
        x_nullcline=x_nullclines,
        y_nullcline=y_nullclines,
        **{
          **dict(
            delta_nullcline=100,
            trail_style='-',
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

    if labels:
      utp.set(
        ax,
        title=(rf'$\mathtt{{strength}}$ = $\mathdefault{{{strength:g}}}$', dict(loc='right')),
      )


def plot_period_response_curve(
  data: ConstantData,
  strengths: list[float] | None = None,
  bar_kws: dict[str, T.Any] = {},
  cmap: str = 'coolwarm',
  duration_states: bool = True,
  duration_total: bool = True,
  duty_cycle: bool = True,
  active_only: bool = True,
  quiet_only: bool = True,
  axs: T.Any = None,
):
  strengths = sorted(strengths or data.strengths)
  axis_strengths = utp.axisinfo(strengths, extra_within=[0.], center=0.)
  color_strengths = mpl.cm.ScalarMappable(  # type: ignore
    cmap=cmap,
    norm=mpl.colors.Normalize(*axis_strengths.outer_limits),
  )
  wave_stims = [data.variants[strength].wave_stim for strength in strengths]
  T_actives = np.array([ws.stats.T_active if ws.stats is not None else np.nan for ws in wave_stims])
  T_quiets = np.array([ws.stats.T_quiet if ws.stats is not None else np.nan for ws in wave_stims])
  p_actives = np.array([ws.stats.p_active if ws.stats is not None else np.nan
                        for ws in wave_stims]) * 100
  i_active_only = max([i for i, x in enumerate(T_quiets) if np.isfinite(x)]) + 1
  i_quiet_only = min([i for i, x in enumerate(T_actives) if np.isfinite(x)]) - 1
  bound = max(*T_actives[np.isfinite(T_actives)], *T_quiets[np.isfinite(T_quiets)]) * 1.05
  idxs = np.arange(len(strengths))
  colors = np.array(
    [color_strengths.to_rgba(strength) if strength != 0 else 'k' for strength in strengths],
    dtype=np.dtype('O'),
  )

  if axs is None:
    _, axs = plt.subplots(1, 3, figsize=utp.figsize(1, h=0.2, r=len(strengths)), width_ratios=[21/2, 21/2, 3], sharey=True, layout='constrained')
  idx = 0

  # Plot active/quiet period vs. stimulus strength, left-aligned (response curve).
  ax = axs[idx]
  if duration_states and ax:
    ax.axhline(strengths.index(0.), c='k', lw=0.5, zorder=-1)
    ax.barh(idxs, -T_actives, color=colors, edgecolor=colors, **{**dict(lw=1.), **bar_kws})
    ax.barh(idxs, T_quiets, color='w', edgecolor=colors, **{**dict(lw=1.), **bar_kws})
    ax.plot(-T_actives, idxs, '.-', c='x', lw=1.)
    ax.plot(T_quiets, idxs, '.-', c='x', lw=1.)
    ax.axvline(0, c='k', lw=0.5)
    ax.set_xlim(-bound, bound)  # Set xlim before adjusting xticks.
    ax.set_ylim(-1, len(strengths))  # Set ylim to include non-plotted strengths and padding.
    utp.set(
      ax,
      xlabel='Active period | Quiet period',
      ylabel='Stimulus strength',
      yticks=(idxs, [rf'$\mathdefault{{{strength:g}}}$' for strength in strengths]),
      xformatter=(lambda x, pos: f'{abs(x):g}'),
    )
    if active_only: ax.axhspan(i_active_only, len(strengths), color='r1', lw=0, zorder=-2)
    if quiet_only: ax.axhspan(-1, i_quiet_only, color='b1', lw=0, zorder=-2)
    idx += 1

  # Plot active/quiet period vs. stimulus strength, center-aligned (response curve).
  ax = axs[idx]
  if duration_total and ax:
    ax.axhline(strengths.index(0.), c='k', lw=0.5, zorder=-1)
    ax.barh(idxs, T_actives, color=colors, edgecolor=colors, **{**dict(lw=1.), **bar_kws})
    ax.barh(
      idxs, T_quiets, left=T_actives, color='w', edgecolor=colors, **{
        **dict(lw=1.), **bar_kws
      }
    )
    ax.plot(T_actives, idxs, '.-', c='x', lw=1.)
    ax.plot(T_quiets + T_actives, idxs, '.-', c='x', lw=1.)
    ax.set_xlim(0, 2 * bound)
    utp.set(
      ax,
      xlabel='Active period + Quiet period',
      yticks=(idxs, [rf'$\mathdefault{{{strength:g}}}$' for strength in strengths]),
    )
    if active_only: ax.axhspan(i_active_only, len(strengths), color='r1', lw=0, zorder=-2)
    if quiet_only: ax.axhspan(-1, i_quiet_only, color='b1', lw=0, zorder=-2)
    idx += 1

  # Plot duty cycle vs. stimulus strength (response curve).
  ax = axs[idx]
  if duty_cycle and ax:
    ax.axvline(50, c='k', lw=0.5)
    ax.axhline(strengths.index(0.), c='k', lw=0.5)
    ax.plot(p_actives, idxs, '-', c='x', lw=1.)
    ax.scatter(p_actives, idxs, color=colors, marker='.', linewidths=1., zorder=10)
    utp.set(
      ax,
      xlim=(0, 100),
      xticks=(0, 50, 100),
      xlabel='Duty cycle %',
      ylim=(-1, len(strengths)),
      yticks=(idxs, [rf'$\mathdefault{{{strength:g}}}$' for strength in strengths]),
    )
    if active_only: ax.axhspan(i_active_only, len(strengths), color='r1', lw=0, zorder=-2)
    if quiet_only: ax.axhspan(-1, i_quiet_only, color='b1', lw=0, zorder=-2)


def plot(
  data: ConstantData,
  x_nullcline: T.Callable | None = None,
  y_nullcline: T.Callable | None = None,
  timeseries_kws: dict[str, T.Any] = {},
  phase_kws: dict[str, T.Any] = {},
  baseline: bool = False,
  output_dir: os.PathLike | str = '.',
) -> None:
  output_dir = pathlib.Path(output_dir)
  os.makedirs(output_dir, exist_ok=True)

  utp.setup_plotting()

  # Plot baseline waveform (timeseries, phase portrait).
  plot_baseline(
    data,
    x_nullcline=x_nullcline,
    y_nullcline=y_nullcline,
    timeseries_kws=timeseries_kws,
    phase_kws=phase_kws,
  )
  plt.savefig(output_dir / 'baseline.pdf')
  plt.close()

  # Plot constant-stimulated waveforms (timeseries).
  plot_waveforms(
    data,
    x_nullcline=x_nullcline,
    y_nullcline=y_nullcline,
    timeseries_kws=timeseries_kws,
    phase_kws=phase_kws,
    baseline=baseline,
  )
  plt.savefig(output_dir / 'waveforms.pdf')
  plt.close()

  # Plot period response curve (centered periods, cumulative periods, duty cycle).
  plot_period_response_curve(data)
  plt.savefig(output_dir / 'period_response_curve.pdf')
  plt.close()
