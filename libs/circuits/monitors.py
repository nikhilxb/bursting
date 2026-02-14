"""
This module defines `Monitor` classes, which are nodes that observe and record the state of other
nodes during simulation. Monitors are useful for debugging, but they should usually be omitted when
training in order to reduce computational overhead.
"""
import abc
import dataclasses
import typing as T

import numpy as np

from circuits import nodes, signal
from utils import collections
from utils.torch import buffers


class Monitor(nodes.PrimitiveNode, abc.ABC):
  """A primitive node that observes and records the state of other nodes during simulation."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def populate(self):
    pass


@dataclasses.dataclass
class TimeseriesMonitorSource:
  signal: signal.Signal
  attr: str
  buffer: buffers.DelayBuffer
  meta: collections.AttrDict


class TimeseriesMonitor(Monitor):
  """A monitor that records timeseries data from one or more node sources."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.sources: list[TimeseriesMonitorSource] = []

  def init(self, cfg):
    self.dt = cfg.timestep

  def connect(
    self,
    signal: signal.Signal,
    attr: str = 'value',
    limit: float | None = None,
    **kwargs,
  ):
    """Connects this monitor to record a particular attribute of a source signal. By default, the
    attribute is named 'value'.

    Args:
      signal: Source signal to record.
      attr: Attribute of the source signal to record.
      limit: Time (in milliseconds) to maintain buffer data. If None, the buffer is unbounded.
      **kwargs: Metadata to associate with the connection.
    """
    if not hasattr(signal, attr):
      raise AttributeError(f"{type(signal).__name__} source does not have attribute '{attr}'")
    timesteps = int(np.ceil(limit / self.dt)) if limit is not None else None
    self.sources.append(
      TimeseriesMonitorSource(
        signal=signal,
        attr=attr,
        buffer=buffers.DelayBuffer(timesteps),
        meta=collections.AttrDict(kwargs),
      )
    )
    return self

  def reset(self):
    """Clears the recorded data of all connections."""
    for source in self.sources:
      source.buffer.clear()

  def populate(self):
    """Populates the monitor with current values of all connections."""
    for source in self.sources:
      value = getattr(source.signal, source.attr)
      source.buffer.step(value)
