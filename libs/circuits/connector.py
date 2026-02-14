"""
This module defines `Connector` classes, which model synapses and connectivity motifs.
"""
import dataclasses

import torch

from circuits import signal
from utils import collections
from utils.torch import buffers


@dataclasses.dataclass
class Connection:
  source: signal.Signal
  attr: str
  weight: float | torch.nn.Parameter
  delay: buffers.DelayBuffer
  noise: float
  meta: collections.AttrDict


class Connector:
  """A data structure that connects to `Signal` nodes and integrates their current values into
  an output. It can be viewed as a general way to model synapses and connectivity motifs.

  The current implementation models a chemical synapse through a simple weighted sum of connected
  signals with optional noise and delay. A future implementation may support electrical synapses
  and more complex connectivity patterns (e.g. topographical, probabilistic, multicompartmental).

  As this class is not a node in the circuit tree, any `torch.Tensor`/`torch.nn.Parameter` (e.g.
  connection weights) should be managed externally and attached to a `torch.nn.Module`.
  """
  def __init__(self, parent: torch.nn.Module):
    self.connections: dict[str, Connection] = {}
    self.parent = parent

  def connect(
    self,
    source: signal.Signal,
    *,
    attr: str = 'value',
    weight: float | torch.nn.Parameter = 1.,
    delay: int = 1,
    noise: float = 0.,
    name: str | None = None,
    **kwargs,
  ):
    """Connects a signal source to the connector with a given weight, delay, and/or noise.

    Args:
      source: Source signal node.
      attr: Attribute of the source signal to connect. Default: 'value'.
      weight: Synaptic weight, whcih can be a scalar or a tensor of the same shape as the
        source signal.
      delay: Delay of the connection (in timesteps).
      noise: Gaussian noise standard deviation of the synaptic weight, mean 1.
      name: Name of the connection.
      **kwargs: Metadata to associate with the connection.
    """
    if not 0 <= noise <= 1: raise ValueError(f'Expected 0 <= `noise` <= 1')
    if not delay >= 1: raise ValueError(f'Expected `delay` >= 1')
    x: torch.Tensor = getattr(source, attr)
    connection = Connection(
      source=source,
      attr=attr,
      weight=weight,
      delay=buffers.DelayBuffer(x.shape, delay),
      noise=noise,
      meta=collections.AttrDict(kwargs),
    )
    name = name or str(len(self.connections))
    # In functional mode, torch replaces parameters and buffers registered to any `torch.nn.Module`
    # but not non-`Module` data structures like this `Connector`. Therefore, we must register these
    # to the parent `Module` and must retrieve them on usage.
    if isinstance(weight, torch.Tensor):
      self.parent.register_parameter(f'weight_{name}', weight)
    self.parent.register_buffer(f'buffer_{name}', connection.delay.timesteps, persistent=False)
    self.connections[name] = connection
    return self

  def __len__(self):
    return len(self.connections)

  def reset(self):
    """Resets the connections to zeros."""
    for c in self.connections.values():
      c.delay.fill()

  def integrate(self) -> torch.Tensor:
    """Integrates the current values of all connected signals."""
    total = torch.zeros(1)
    for name, c in self.connections.items():
      c.delay.timesteps = getattr(self.parent, f'buffer_{name}')
      total = total + c.delay.get()
    return total

  def populate(self):
    """Populates the connector with current values of all connected signals."""
    for name, c in self.connections.items():
      x = getattr(c.source, c.attr)
      w = getattr(self.parent, f'weight_{name}', c.weight)
      c.delay.timesteps = getattr(self.parent, f'buffer_{name}')
      if c.noise:
        w = w + torch.randn(getattr(w, 'shape', ())) * c.noise
      x = w * x
      c.delay.step(x)
