"""
This module defines `Generator` classes, which are nodes that generate signals in the absence of
inputs.
"""
import abc

import numpy as np
import torch

from circuits import signal


class Generator(signal.Signal, abc.ABC):
  """A primitive node that generates signals in the absence of inputs."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def step(self):
    """Advances the generator to the next signal value."""
    pass


class PeriodicGenerator(Generator):
  """A generator that produces a periodic signal by cycling through the provided pattern."""
  def __init__(
    self,
    pattern: torch.Tensor,
    **kwargs,
  ):
    """
    Args:
      pattern: Pattern to cycle through, shape (..., N). The last dimension of the pattern tensor
      is the dimension that is cycled through, advancing on each `step()`.
    """
    if not pattern.ndim >= 1: raise ValueError(f'Expected `pattern` to be at least 1D')
    super().__init__(**kwargs)
    self.pattern: torch.Tensor = pattern

  def reset(self):
    """Resets the generator to the first value in the pattern."""
    self.value = self.pattern[..., 0]
    self.index = 0

  def step(self):
    """Advances the generator to the next value in the pattern."""
    self.index = (self.index + 1) % self.pattern.shape[-1]
    self.value = self.pattern[..., self.index, None]


class SquareGenerator(PeriodicGenerator):
  """A generator that produces a square wave signal."""
  def __init__(
    self,
    period: int = 2,
    duration: int = 1,
    phase: int = 0,
    low: float = 0.,
    high: float = 1.,
    **kwargs,
  ):
    """
    Args:
      period: Period of the waveform (in timesteps).
      duration: Duration of the high phase (in timesteps).
      phase: Phase offset of the high phase (in timesteps).
      low: Signal value during the low phase.
      high: Signal value during the high phase.
    """
    if not period >= 1: raise ValueError(f'Expected `period` >= 1')
    if not duration >= 0: raise ValueError(f'Expected `duration` >= 0')
    pattern = torch.full((period,), low)
    pattern[phase:phase + duration] = high
    super().__init__(pattern, **kwargs)


class SineGenerator(PeriodicGenerator):
  """A generator that produces a sine wave signal."""
  def __init__(
    self,
    period: int = 4,
    phase: float = 0.,
    amplitude: float = 1.,
    shift: float = 0.,
    **kwargs,
  ):
    """
    Args:
      period: Period of the waveform (in timesteps).
      phase: Phase offset of the cycle start (in timesteps).
      amplitude: Signal value range from lowest to highest point.
      shift: Signal value midpoint.
    """
    if period < 1: raise ValueError(f'Expected `period` >= 1')
    pattern = amplitude * torch.sin(2 * np.pi / period * (torch.arange(0., period) - phase)) + shift
    super().__init__(pattern, **kwargs)
