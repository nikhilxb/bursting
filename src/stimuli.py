import typing as T

import numpy as np

Stimulus = T.Callable[[float, float], float]


def constant(value: float = 1.) -> Stimulus:
  def stimulus(t: float, T: float) -> float:
    return value

  return stimulus


def fraction(
  high: float = 1.,
  low: float = 0.,
  fraction: float = 0.5,
) -> Stimulus:
  start = (1 - np.clip(fraction, 0, 1)) / 2
  end = 1 - start

  def stimulus(t: float, T: float) -> float:
    return high if start * T <= t < end * T else low

  return stimulus


def pulse(
  high: float = 1.,
  low: float = 0.,
  width: float = 3.,
  start: float = 0.,
) -> Stimulus:
  def stimulus(t: float, T: float) -> float:
    return high if start <= t < start + width else low

  return stimulus


def periodic(
  high: float = 1.,
  low: float = 0.,
  width: float = 3.,
  period: float = 10.,
  start: float = 0.,
  count: int | None = None,
) -> Stimulus:
  if count is None:
    end = float('inf')
  elif count == 0:
    end = start
  else:
    end = start + width + period * (count - 1)

  def stimulus(t: float, T: float) -> float:
    return high if start <= t < end and 0 <= (t - start) % period < width else low

  return stimulus


def square(
  high: float = 1.,
  low: float = 0.,
  duty: float = 0.5,
  period: float = 10.,
  start: float = 0.
) -> Stimulus:
  def stimulus(t: float, T: float) -> float:
    return high if start <= t and 0 <= (t - start) % period < period * duty else low

  return stimulus


def sine(
  high: float = 1.,
  low: float = 0.,
  period: float = 10.,
) -> Stimulus:
  mid = (high + low) / 2
  amp = abs(high - mid)

  def stimulus(t: float, T: float) -> float:
    return amp * np.sin(2 * np.pi / period * t) + mid

  return stimulus


def steps(
  start: float = 0.,
  end: float = 1.,
  count: int = 5,
  width: float | None = None,
):
  assert width is None or width > 0
  if count <= 1: return constant(start)

  height = (end - start) / (count - 1)
  values = [start + i * height for i in range(count)]

  def stimulus(t: float, T: float) -> float:
    w = width or T / count
    return values[int(t // w) % count]

  return stimulus


PRESETS: dict[str, Stimulus] = {
  'constant': constant(),
  'half': fraction(),
  'periodic10': periodic(period=10),
  'periodic20': periodic(period=20),
  'periodic50': periodic(period=50),
  'square10': square(period=10),
  'square20': square(period=20),
  'square50': square(period=50),
  'square10*': square(period=10, low=-1),
  'square20*': square(period=20, low=-1),
  'square50*': square(period=50, low=-1),
  'sine10': sine(period=10),
  'sine20': sine(period=20),
  'sine50': sine(period=50),
  'sine10*': sine(period=10, low=-1),
  'sine20*': sine(period=20, low=-1),
  'sine50*': sine(period=50, low=-1),
  'neg_to_pos': steps(start=-1, end=1),
  'pos_to_neg': steps(start=1, end=-1),
  'neg_pos_neg': square(low=-1, high=1, period=2000, start=1000),
  'pulse1000_zero_one': pulse(low=0, high=1, width=1000, start=2000),
  'pulse1000_half_one': pulse(low=0.5, high=1, width=1000, start=2000),
  'pulse1000_one_zero': pulse(low=1, high=0, width=1000, start=2000),
  'pulse1000_one_half': pulse(low=1, high=0.5, width=1000, start=2000),
  'pulse1000_neg_pos': pulse(low=-1, high=1, width=1000, start=2000),
}
