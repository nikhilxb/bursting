import typing as T

import torch

ConstraintBounds = tuple[int | float | None, int | float | None]
ConstraintFn = T.Callable[[torch.Tensor], torch.Tensor]


class ConstrainedParameter(torch.nn.Parameter):
  """A parameter with a constraint function that applies the constraint on construction and
  modification."""
  constraint: ConstraintFn | None

  def __new__(
    cls,
    data: torch.Tensor | None = None,
    requires_grad: bool = True,
    *,
    constraint: ConstraintBounds | ConstraintFn | None = None,
  ):
    if isinstance(constraint, (tuple, list)):
      lower, upper = constraint

      def clamp(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_(min=lower, max=upper)

      constraint = clamp
    data = constraint(data) if constraint is not None and data is not None else data
    obj = super().__new__(cls, data, requires_grad)
    obj.constraint = constraint
    return obj

  def __setattr__(self, name: str, value: T.Any):
    if name == 'data' and self.constraint is not None:
      with torch.no_grad():
        value = self.constraint(value)
    super().__setattr__(name, value)

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    if self.constraint is not None:
      with torch.no_grad():
        self.data[:] = self.constraint(self.data)

  def __repr__(self):
    return "ConstrainedParameter containing:\n" + repr(self.data)
