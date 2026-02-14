import itertools
import logging

import numpy as np
import pandas as pd

from . import models, stimuli

log = logging.getLogger(__name__)


def weights_generator(number: int, spec: str = '', default: float = 0.):
  weights = ~np.eye(number, dtype=bool) * default
  for group in (group.strip().split() for group in spec.strip().split(';')):
    if len(group) == 0: continue
    pre, post, weight = group
    pre = int(pre)
    post = int(post)
    weight = float(weight)
    if pre < number and post < number:
      weights[pre, post] = weight
    else:
      log.debug(f'Ignoring weight: {pre}->{post} = {weight}')
  return weights


def biases_generator(number: int, spec: str = '', default: float = 0.):
  biases = np.ones(number) * default
  for idx, bias in enumerate(spec.strip().split()):
    bias = float(bias)
    if idx < number:
      biases[idx] = bias
    else:
      log.debug(f'Ignoring bias: {idx} = {bias}')
  return biases


def cartesian(
  model: models.CartesianModel,
  T: float = 100,
  dt: float = 1,
  v_init: float = 0.,
  a_init: float = 0.,
  number: int = 1,
  weights: str = '',
  biases: str = '',
  stimulus: stimuli.Stimulus | str | None = None,
  strength: float = 1.,
  seed: int = 0,
):
  np.random.seed(seed)
  if stimulus is None:
    stimulus = stimuli.constant(0)
  elif isinstance(stimulus, str):
    stimulus = stimuli.PRESETS[stimulus]
  w = weights_generator(number, weights)
  b = biases_generator(number, biases)
  v = np.zeros(number) + v_init
  a = np.zeros(number) + a_init
  i = strength * stimulus(0, T)
  data = [
    tuple(itertools.chain(
      [0],
      *[(v[n], a[n], model.y(v[n], a[n], i), i) for n in range(number)],
    ))
  ]  # (t, v, a, y, i)
  for t in np.arange(dt, T + dt, dt):
    i = model.y(v, a) @ w + b + strength * stimulus(t, T)
    v, a = model.step(v, a, i, dt)
    data.append(
      tuple(
        itertools.chain(
          [t],
          *[(v[n], a[n], model.y(v[n], a[n], i[n]), i[n]) for n in range(number)],
        )
      )
    )
  return pd.DataFrame(
    data,
    columns=tuple(
      itertools.chain(
        ['t'],
        *[(f'v_{n}', f'a_{n}', f'y_{n}', f'i_{n}') for n in range(number)],
      )
    )
  )


def polar(
  model: models.PolarModel,
  T: float = 100,
  dt: float = 1,
  r_init: float = 0.,
  p_init: float = 0.,
  number: int = 1,
  weights: str = '',
  biases: str = '',
  stimulus: stimuli.Stimulus | str | None = None,
  strength: float = 1.,
  seed: int = 0,
):
  np.random.seed(seed)
  if stimulus is None:
    stimulus = stimuli.constant(0)
  elif isinstance(stimulus, str):
    stimulus = stimuli.PRESETS[stimulus]
  w = weights_generator(number, weights)
  b = biases_generator(number, biases)
  r = np.zeros(number) + r_init
  p = np.zeros(number) + p_init
  i = strength * stimulus(0, T)
  data = [
    tuple(
      itertools.chain(
        [0],
        *[(r[n], p[n], model.v(r[n], p[n]), model.a(r[n], p[n]), model.y(r[n], p[n], i), i)
          for n in range(number)],
      )
    )
  ]  # (t, r, p, v, a, y, i)
  for t in np.arange(dt, T + dt, dt):
    i = model.y(r, p) @ w + b + strength * stimulus(t, T)
    r, p = model.step(r, p, i, dt)
    data.append(
      tuple(
        itertools.chain(
          [t],
          *[(r[n], p[n], model.v(r[n], p[n]), model.a(r[n], p[n]), model.y(r[n], p[n], i[n]), i[n])
            for n in range(number)],
        )
      )
    )
  return pd.DataFrame(
    data,
    columns=tuple(
      itertools.chain(
        ['t'],
        *[(f'r_{n}', f'p_{n}', f'v_{n}', f'a_{n}', f'y_{n}', f'i_{n}') for n in range(number)],
      )
    )
  )
