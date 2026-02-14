import typing as T

import numpy as np
import numpy.typing as npt

from . import base

InterpolateFn = T.Callable[
  [np.ndarray | float, np.ndarray | float, np.ndarray | float],
  np.ndarray | float,
]


def linear(
  start: np.ndarray | float,
  end: np.ndarray | float,
  z: np.ndarray | float,
) -> np.ndarray | float:
  """Interpolate linearly, such that z = 0 maps to `start` and z = 1 maps to `end`."""
  return start * (1 - z) + end * z


def logarithmic(
  start: np.ndarray | float,
  end: np.ndarray | float,
  z: np.ndarray | float,
  base: np.ndarray | float = 2,
) -> np.ndarray | float:
  """Interpolate logarithmically, such that z = 0 maps to `start` and z = 1 maps to `end`."""
  assert base > 0
  return start + (end - start) * np.log((base - 1) * z + 1) / np.log(base)


def clipnorm(
  start: np.ndarray | float,
  end: np.ndarray | float,
  z: np.ndarray | float,
) -> np.ndarray | float:
  """Normalize linearly, such that z = `start` maps to 0 and z = `end` maps to 1."""
  return np.clip((z - start) / (end - start), 0, 1)


def y_rectangular(
  v: np.ndarray,
  a: np.ndarray,
  active_bound: np.ndarray | float = 0.,
  quiet_bound: np.ndarray | float = 1.,
  active: np.ndarray | float = 1.,
  quiet: np.ndarray | float = 0.,
  **kwargs,
) -> np.ndarray:
  return (v > 0) * active + (v < 0) * quiet


def y_linear(
  v: np.ndarray,
  a: np.ndarray,
  active_bound: np.ndarray | float = 0.,
  quiet_bound: np.ndarray | float = 1.,
  active_start: np.ndarray | float = 1.0,
  active_end: np.ndarray | float = 0.5,
  quiet: np.ndarray | float = 0.,
  **kwargs,
) -> np.ndarray:
  return ((v > 0) * linear(active_end, active_start, clipnorm(active_bound, quiet_bound, a)) +
          (v < 0) * quiet)


def y_logarithmic(
  v: np.ndarray,
  a: np.ndarray,
  active_bound: np.ndarray | float = 0.,
  quiet_bound: np.ndarray | float = 1.,
  active_start: np.ndarray | float = 1.0,
  active_end: np.ndarray | float = 0.5,
  quiet: np.ndarray | float = 0.,
  base: float = 2.,
  **kwargs,
) -> np.ndarray:
  return ((v > 0) *
          logarithmic(active_end, active_start, clipnorm(active_bound, quiet_bound, a), base=base) +
          (v < 0) * quiet)


class SimplifiedOscillator(base.CartesianModelWithNullclines):
  def __init__(
    self,
    variant: T.Literal['symmetric'] = 'symmetric',
    activation: T.Literal[
      'rectangular',
      'linear',
      'logarithmic',
      'biolike',
    ] | T.Callable = 'biolike',
    B: float = 0.,  # Intrinsic bias.
    T_active: float = 400.,  # Active period time (ms) at i = 0.
    T_quiet: float = 1000.,  # Quiet period time (ms) at i = 0.
    T_adaptation: float = 2400.,  # Adaptation time (ms) at i = 0.
    k_active_pos: float = 0.5,  # Active scale at i = +1.
    k_quiet_pos: float = 0.1,  # Quiet scale at i = +1.
    k_active_neg: float = 0.7,  # Active scale at i = -1.
    k_quiet_neg: float = 1.3,  # Quiet scale at i = -1.
    k_adaptation_active_pos: float = 1.,  # Adaptation scale in active period at i = +1.
    k_adaptation_quiet_pos: float = 1.1,  # Adaptation scale in quiet period at i = +1.
    k_adaptation_active_neg: float = 1.1,  # Adaptation scale in active period at i = -1.
    k_adaptation_quiet_neg: float = 0.9,  # Adaptation scale in quiet period at i = -1.
    x_active_stable: float = 1.,  # Input i > 0 for active stable point (tonic).
    x_quiet_stable: float = 0.,  # Input i < 0 for quiet stable point (quiescent).
    t_active_stable: float = 0.,  # Time margin (ms) for active stable point clipping.
    t_quiet_stable: float = 0.,  # Time margin (ms) for quiet stable point clipping.
    d_active: float = 0.,  # Delay fraction of active period.
    d_quiet: float = 0.1,  # Delay fraction of quiet period.
    noise: float = 0.,  # Standard deviation of multiplicative Gaussian noise (mean 1).
  ):
    super().__init__()
    assert variant in ('symmetric',)
    assert -1 <= B <= 1
    assert 0 < T_active
    assert 0 < T_quiet
    assert 0 < T_adaptation
    assert 0 < k_active_pos
    assert 0 < k_quiet_pos
    assert 0 < k_active_neg
    assert 0 < k_quiet_neg
    assert 0 < k_adaptation_active_pos
    assert 0 < k_adaptation_quiet_pos
    assert 0 < k_adaptation_active_neg
    assert 0 < k_adaptation_quiet_neg
    assert x_quiet_stable <= 0 <= x_active_stable
    assert 0 <= t_active_stable
    assert 0 <= t_quiet_stable
    assert 0 <= d_active < 1
    assert 0 <= d_quiet < 1
    assert 0 <= noise

    self.variant = variant
    self.activation = activation
    self.B = B
    self.T_active = T_active
    self.T_quiet = T_quiet
    self.T_adaptation = T_adaptation
    self.k_active_pos = k_active_pos
    self.k_quiet_pos = k_quiet_pos
    self.k_active_neg = k_active_neg
    self.k_quiet_neg = k_quiet_neg
    self.k_adaptation_active_pos = k_adaptation_active_pos
    self.k_adaptation_quiet_pos = k_adaptation_quiet_pos
    self.k_adaptation_active_neg = k_adaptation_active_neg
    self.k_adaptation_quiet_neg = k_adaptation_quiet_neg
    self.x_active_stable = x_active_stable
    self.x_quiet_stable = x_quiet_stable
    self.t_active_stable = t_active_stable
    self.t_quiet_stable = t_quiet_stable
    self.d_active = d_active
    self.d_quiet = d_quiet
    self.noise = noise

    match self.variant:
      case 'symmetric':
        Ta = 4 * T_active / T_adaptation
        Tq = 4 * T_quiet / T_adaptation
        Ta_pos = Ta * k_active_pos / k_adaptation_active_pos
        Tq_pos = Tq * k_quiet_pos / k_adaptation_quiet_pos
        Ta_neg = Ta * k_active_neg / k_adaptation_active_neg
        Tq_neg = Tq * k_quiet_neg / k_adaptation_quiet_neg
        self.active_bound_zero = (1 - np.exp(Tq)) / (1 - np.exp(Ta + Tq))
        self.quiet_bound_zero = self.active_bound_zero * np.exp(Ta)
        self.active_bound_pos = (1 - np.exp(Tq_pos)) / (1 - np.exp(Ta_pos + Tq_pos))
        self.quiet_bound_pos = self.active_bound_pos * np.exp(Ta_pos)
        self.active_bound_neg = (1 - np.exp(Tq_neg)) / (1 - np.exp(Ta_neg + Tq_neg))
        self.quiet_bound_neg = self.active_bound_neg * np.exp(Ta_neg)

  def step(self, v: npt.NDArray[np.floating], a, i, dt):
    x = np.round(i + self.B, 10)
    pos = np.clip(x, 0, 1)
    neg = np.clip(-x, 0, 1)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    match self.variant:
      case 'symmetric':
        k = 4 / self.T_adaptation
        k_active = k / (
          linear(1, self.k_adaptation_active_pos, pos) * pos_mask +
          linear(1, self.k_adaptation_active_neg, neg) * neg_mask
        )
        k_quiet = k / (
          linear(1, self.k_adaptation_quiet_pos, pos) * pos_mask +
          linear(1, self.k_adaptation_quiet_neg, neg) * neg_mask
        )
        active_bound = (
          linear(self.active_bound_zero, self.active_bound_pos, pos) * pos_mask +
          linear(self.active_bound_zero, self.active_bound_neg, neg) * neg_mask
        )
        quiet_bound = (
          linear(self.quiet_bound_zero, self.quiet_bound_pos, pos) * pos_mask +
          linear(self.quiet_bound_zero, self.quiet_bound_neg, neg) * neg_mask
        )

    # Adaptation constraints: Clip to bounds
    #   (1) before integration, if within stable point time margin --> moves to new stable point
    #   (2) after integration, if just passed the bound on this timestep --> prevents overshoot
    # Clipping ensures the stable point is exactly at the bound elbow and any overshoot should
    # begin a state switch (e.g. for post-inhibitory rebound).
    active_bound_with_margin = active_bound * np.exp(-self.t_active_stable * k_active)
    quiet_bound_with_margin = 1 + (quiet_bound - 1) * np.exp(-self.t_quiet_stable * k_quiet)
    update_active_stable = (v > 0) & (active_bound_with_margin <= a) & (a <= active_bound)
    update_quiet_stable = (v < 0) & (quiet_bound <= a) & (a <= quiet_bound_with_margin)
    clip = update_active_stable | update_quiet_stable
    a[clip] = np.clip(a, active_bound, quiet_bound)[clip]
    # Forward-Euler discretization.
    _, da = self.dv_da(v, a, i)
    a_old = a.copy()
    a = a + da * dt
    passed_active_bound = (v > 0) & (a <= active_bound) & (a_old >= active_bound)
    passed_quiet_bound = (v < 0) & (a >= quiet_bound) & (a_old <= quiet_bound)
    clip = passed_active_bound | passed_quiet_bound
    a[clip] = np.clip(a, active_bound, quiet_bound)[clip]

    # Recalculate derivatives to use most recent adaptation. Otherwise, the voltage derivative
    # is incorrect.

    # Voltage constraints: Flip if adaptation at/passed bounds and voltage at/passed threshold.
    # Forward-Euler discretization.
    dv, _ = self.dv_da(v, a, i)
    v = v + dv * dt
    v = np.clip(v, -1, +1)
    v[(a <= active_bound) & (v <= 0)] = -1
    v[(a >= quiet_bound) & (v >= 0)] = +1

    return v, a

  def dv_da(self, v, a, i=0):
    x = np.round(i + self.B, 10)
    pos = np.clip(x, 0, 1)
    neg = np.clip(-x, 0, 1)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    match self.variant:
      case 'symmetric':
        k = 4 / self.T_adaptation
        k_active = k / (
          linear(1, self.k_adaptation_active_pos, pos) * pos_mask +
          linear(1, self.k_adaptation_active_neg, neg) * neg_mask
        )
        k_quiet = k / (
          linear(1, self.k_adaptation_quiet_pos, pos) * pos_mask +
          linear(1, self.k_adaptation_quiet_neg, neg) * neg_mask
        )
        active_bound = (
          linear(self.active_bound_zero, self.active_bound_pos, pos) * pos_mask +
          linear(self.active_bound_zero, self.active_bound_neg, neg) * neg_mask
        )
        quiet_bound = (
          linear(self.quiet_bound_zero, self.quiet_bound_pos, pos) * pos_mask +
          linear(self.quiet_bound_zero, self.quiet_bound_neg, neg) * neg_mask
        )

        # Noise is multiplicative scalar with mean 1.
        noise = np.clip(1 + self.noise * np.random.normal(size=v.shape), 0, None)

        # Adaptation derivative: Moves when within bounds.
        # da = (
        #   if [inside active bound], [decrease a] [at adaptation timescale] [rescaled for delay]
        #   if [inside quiet bound], [increase a] [at adaptation timescale] [rescaled for delay]
        # ) * [speed-up/slow-down with noise]
        inside_active_bound = (v > 0) & (a > active_bound)
        inside_quiet_bound = (v < 0) & (a < quiet_bound)
        da = (
          inside_active_bound * (0 - a) * k_active / (1 - self.d_active) + inside_quiet_bound *
          (1 - a) * k_quiet / (1 - self.d_quiet)
        ) * noise

        # Voltage derivative: Moves to elbow when inside bounds, then moves either towards zero for
        # state switch or towards elbow if input enough for a stable point (tonic or suppressed).
        # The rate is inversely proportional to the delay time (phase time * delay fraction).
        # The direction is outwards (to +1 or -1) for a stable point, or inwards (to 0) otherwise.
        active_delay = self.d_active * (np.log(quiet_bound) - np.log(active_bound)) / k_active
        quiet_delay = self.d_quiet * (np.log(1 - active_bound) - np.log(1 - quiet_bound)) / k_quiet
        active_rate = 1 / np.maximum(active_delay, 1)  # Avoid division by zero.
        quiet_rate = 1 / np.maximum(quiet_delay, 1)
        toward_active_stable = (v > 0) & (a == active_bound) & (x > self.x_active_stable)
        toward_quiet_stable = (v < 0) & (a == quiet_bound) & (x < self.x_quiet_stable)
        active_direction = np.sign(+1 * (inside_active_bound | toward_active_stable) - v)
        quiet_direction = np.sign(-1 * (inside_quiet_bound | toward_quiet_stable) - v)
        dv = ((v > 0) * active_rate * active_direction +
              (v < 0) * quiet_rate * quiet_direction) * noise

    return dv, da

  def y(self, v, a, i=0):
    x = np.round(i + self.B, 10)
    pos = np.clip(x, 0, 1)
    neg = np.clip(-x, 0, 1)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    active_bound = (
      linear(self.active_bound_zero, self.active_bound_pos, pos) * pos_mask +
      linear(self.active_bound_zero, self.active_bound_neg, neg) * neg_mask
    )
    quiet_bound = (
      linear(self.quiet_bound_zero, self.quiet_bound_pos, pos) * pos_mask +
      linear(self.quiet_bound_zero, self.quiet_bound_neg, neg) * neg_mask
    )
    match self.activation:
      case 'rectangular':
        return y_rectangular(v, a, active_bound=active_bound, quiet_bound=quiet_bound)
      case 'linear':
        return y_linear(v, a, active_bound=active_bound, quiet_bound=quiet_bound)
      case 'logarithmic':
        return y_logarithmic(v, a, active_bound=active_bound, quiet_bound=quiet_bound)
      case 'biolike':
        return y_logarithmic(
          v,
          a,
          active_bound=self.active_bound_pos,
          quiet_bound=self.quiet_bound_zero,
          active_start=linear(0.85, 0.90, clipnorm(0.0, 1.0, pos)),
          active_end=linear(0.05, 0.2, clipnorm(0.0, 1.0, pos)),
          quiet=linear(0, 0.1, clipnorm(0.8, 1.0, pos)),
          base=20,
        )
      case _:
        if callable(self.activation):
          return self.activation(v, a, active_bound=active_bound, quiet_bound=quiet_bound, x=x)
        raise ValueError(f'Invalid activation: {self.activation}')

  def v_nullcline(self, v, a, i=0):
    # Construct nullcline in vertical and horizontal pieces.
    x = np.round(i + self.B, 10)
    pos = np.clip(x, 0, 1)
    neg = np.clip(-x, 0, 1)

    match self.variant:
      case 'symmetric':
        active_bound = (
          linear(self.active_bound_zero, self.active_bound_pos, pos) * (x >= 0) +
          linear(self.active_bound_zero, self.active_bound_neg, neg) * (x < 0)
        )
        quiet_bound = (
          linear(self.quiet_bound_zero, self.quiet_bound_pos, pos) * (x >= 0) +
          linear(self.quiet_bound_zero, self.quiet_bound_neg, neg) * (x < 0)
        )

      case _:
        raise ValueError(f'Invalid variant: {self.variant}')

    v_quiet_vertical = v[(-1 <= v) & (v <= 0)]
    a_quiet_horizontal = a[a <= quiet_bound]
    v_active_vertical = v[(0 <= v) & (v <= 1)]
    a_active_horizontal = a[active_bound <= a]

    # Must order in ascending order as pieces will be connected.
    return np.concatenate([
      np.full_like(a_quiet_horizontal, -1),
      v_quiet_vertical,
      v_active_vertical,
      np.full_like(a_active_horizontal, 1),
    ]), np.concatenate([
      a_quiet_horizontal,
      np.full_like(v_quiet_vertical, quiet_bound),
      np.full_like(v_active_vertical, active_bound),
      a_active_horizontal,
    ])

  def a_nullcline(self, v, a, i=0):
    return np.zeros_like(v), a
