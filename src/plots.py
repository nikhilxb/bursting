import bisect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.plot as utp


def plot_timeseries(
  df: pd.DataFrame,
  t='t',
  v='v',
  a='a',
  y='y',
  i='i',
  c='C0',
  preprocess=None,
  clip_on=True,
  xlim=(None, None),
  ylim=(-1, 1),
  xref=None,
  yref=[0],
  xlabel=None,
  ax=None,
):
  """Plot timeseries.
    df: DataFrame with columns: (t, v, a, y).
    """
  if ax is None:
    _, ax = plt.subplots(figsize=utp.figsize(1, 1 / 3))

  if preprocess:
    df = preprocess(df)

  # Plot reference lines.
  if xref is not None:
    for xcoord in xref:
      ax.vlines(
        xcoord, ylim[0] or min(df[v]), ylim[1] or max(df[v]), colors='k', lw=0.5, ls='-'
      )  # type: ignore
  if yref is not None:
    for ycoord in yref:
      ax.hlines(
        ycoord, xlim[0] or min(df[t]), xlim[1] or max(df[t]), colors='k', lw=0.5, ls='-'
      )  # type: ignore

  # Plot trajectories.
  if y:
    ax.plot(
      df[t],
      df[y],
      label=f'Activation ${y}$',
      c=utp.color(c, l=30),
      lw=1.5,
      zorder=2,
      clip_on=clip_on,
    )
  if v: ax.plot(
      df[t],
      df[v],
      label=f'Voltage ${v}$',
      c=c,
      lw=1,
      zorder=4,
      clip_on=clip_on,
  )
  if a:
    ax.plot(
      df[t],
      df[a],
      label=f'Adaptation ${a}$',
      c=utp.color(c, l=85),
      lw=1,
      zorder=3,
      clip_on=clip_on,
    )
  if i: ax.plot(
      df[t],
      df[i],
      label=f'Input ${i}$',
      c='0.75',
      lw=1,
      zorder=1,
      clip_on=clip_on,
  )

  utp.set(
    ax,
    margins=(0, 1),
    xlim=xlim,
    ylim=ylim,
    xlabel=f'Time ${t}$' if xlabel is None else xlabel,
    legend=(True, dict(loc='lower right', bbox_to_anchor=(1, 1), ncol=4)),
  )


def plot_phase_portrait(
  df,
  t='t',
  x='v_0',
  y='a_0',
  preprocess=None,
  initial=True,
  initial_kws={},
  trail=True,
  trail_style='.-',
  trail_kws={},
  final=True,
  final_kws={},
  xref=[],
  yref=[],
  xref_kws={},
  yref_kws={},
  dx_dy=None,
  x_nullcline=None,
  y_nullcline=None,
  x_nullcline_kws={},
  y_nullcline_kws={},
  margins=0,
  flip=True,
  xlim=(-1, 1),
  ylim=(-1, 1),
  delta_grid=15,
  delta_nullcline=200,
  legend=True,
  xlabel=None,
  ylabel=None,
  ax=None,
):
  """Plot phase portrait.
    df: DataFrame with columns: (t, x, y).
    """
  if ax is None:
    _, ax = plt.subplots(figsize=utp.figsize(1 / 2, 1))

  if preprocess:
    df = preprocess(df)

  xmin0, xmax0 = xlim
  ymin0, ymax0 = ylim
  if flip:
    x, y = y, x
    xref, yref = yref, xref
    xlim, ylim = ylim, xlim
    xlabel, ylabel = ylabel, xlabel
    xref_kws, yref_kws = yref_kws, xref_kws
  xmin, xmax = xlim
  ymin, ymax = ylim

  # Plot reference lines.
  if xref is not None:
    for xcoord in xref:
      ax.vlines(
        xcoord, ymin, ymax, **{
          **dict(colors='k', lw=0.5, ls='-', zorder=10), **xref_kws
        }
      )  # type: ignore
  if yref is not None:
    for ycoord in yref:
      ax.hlines(
        ycoord, xmin, xmax, **{
          **dict(colors='k', lw=0.5, ls='-', zorder=10), **yref_kws
        }
      )  # type: ignore

  # Plot vector field.
  xgrid, ygrid = np.meshgrid(np.linspace(xmin0, xmax0, delta_grid), np.linspace(ymin0, ymax0, delta_grid))
  if dx_dy:
    dxs, dys = dx_dy(xgrid, ygrid)
    if flip: xgrid, ygrid, dxs, dys = ygrid, xgrid, dys, dxs
    ax.quiver(xgrid, ygrid, dxs, dys, color='xl', zorder=0)

  # Plot nullclines.
  xarr = np.linspace(xmin0, xmax0, delta_nullcline)
  yarr = np.linspace(ymin0, ymax0, delta_nullcline)
  if x_nullcline:
    if not isinstance(x_nullcline, (list, tuple)):
      x_nullcline = [x_nullcline]
    for i, x_null in enumerate(x_nullcline):
      xy = x_null(xarr, yarr)
      if xy is not None:
        xs, ys = xy
        if flip: xs, ys = ys, xs
        ax.plot(
          xs,
          ys,
          **{
            **dict(label=f'Nullcline ${x}$', c='k', lw=1, ls='-', zorder=4),
            **(x_nullcline_kws if isinstance(x_nullcline_kws, dict) else x_nullcline_kws[i]),
          }
        )
  if y_nullcline:
    if not isinstance(y_nullcline, (list, tuple)):
      y_nullcline = [y_nullcline]
    for i, y_null in enumerate(y_nullcline):
      xy = y_null(xarr, yarr)
      if xy is not None:
        xs, ys = xy
        if flip: xs, ys = ys, xs
        ax.plot(
          xs,
          ys,
          **{
            **dict(label=f'Nullcline ${y}$', c='xl', lw=1, ls='-', zorder=3),
            **(y_nullcline_kws if isinstance(y_nullcline_kws, dict) else y_nullcline_kws[i]),
          }
        )

  # Plot trajectories.
  if initial:
    ax.plot([df[x].iloc[0]], [df[y].iloc[0]],
            **{
              **dict(marker='x', c='k', zorder=5), **initial_kws
            })
  if trail > 0:
    cutoff = 0 if trail is True else bisect.bisect_left(df[t], df[t].iloc[-1] - trail)
    ax.plot(
      df[cutoff:][x],
      df[cutoff:][y],
      trail_style,
      zorder=6,
      **{
        **dict(c='k', lw=1.5, label=f'Trajectory (${x}$,${y}$)'), **trail_kws
      }
    )
  if final:
    ax.plot([df[x].iloc[-1]], [df[y].iloc[-1]],
            **{
              **dict(marker='o', c='k', zorder=7), **final_kws
            })

  utp.set(ax, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
  utp.set(ax, margins=margins)
  if legend and (x_nullcline or y_nullcline or trail):
    utp.set(ax, legend=(True, dict(loc='lower right', bbox_to_anchor=(1, 1), ncol=3)))
