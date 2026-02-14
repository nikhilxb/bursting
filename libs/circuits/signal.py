"""
This defines the `Signal` class, which is a primitive node that represents a tensor-valued
signal.
"""
import torch

from circuits import nodes


class Signal(nodes.PrimitiveNode):
  """A primitive node that represents a tensor-valued signal. It can be connected to other nodes
  that consume signals through the common `value` attribute. Many nodes subclass this and implement
  distinct internal logic, but it is useful to directly instantiate this class for input from code
  external to the circuits library (e.g. writing environment observations).

  Attributes:
    value (torch.Tensor): Current tensor value of the signal.
  """
  def __init__(self, value: torch.Tensor | float = 0., **kwargs):
    super().__init__(**kwargs)
    # Allows pointer to the same tensor.
    tensor = torch.as_tensor(value, dtype=torch.float)
    if tensor.ndim == 0: tensor = tensor.unsqueeze(0)
    self.value = torch.nn.Buffer(tensor, persistent=False)
    """Current tensor value of the signal."""
