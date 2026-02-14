import typing as T

import numpy as np


def floatrange(start: float, end: float, step: float, precision: int = 10) -> T.Iterable[float]:
  """Numerically stable variant of `numpy.arange(start, end, step)` that includes both endpoints."""
  # Include both endpoints by adding step. Convert -0 to +0.
  # Round to avoid floating-point errors, and convert `np.float64` to `float`.
  return (round(x, precision) or 0. for x in np.arange(start, end + step, step))


@T.overload
def select_every(
  values: T.Sequence[float] | np.ndarray,
  separation: float,
  *,
  idxs: T.Literal[False] = ...,
  precision: int = ...,
) -> list[float]:
  ...


@T.overload
def select_every(
  values: T.Sequence[float] | np.ndarray,
  separation: float,
  *,
  idxs: T.Literal[True],
  precision: int = ...,
) -> tuple[list[float], list[int]]:
  ...


@T.overload
def select_every(
  values: T.Sequence[float] | np.ndarray,
  separation: float,
  *,
  idxs: bool = ...,
  precision: int = ...,
) -> list[float] | tuple[list[float], list[int]]:
  ...


def select_every(
  values: T.Sequence[float] | np.ndarray,
  separation: float,
  *,
  idxs: bool = False,
  precision: int = 10,
) -> list[float] | tuple[list[float], list[int]]:
  """Selects from a sorted sequence such that consecutive values are separated in magnitude by at
  least `separation`."""
  # Need to be careful about floating-point error. A for-loop with rounding was straightforward
  # to achieve desired behavior, compared to numpy calculations that introduced errors resulting in
  # dropped values.
  values = [float(x) for x in values]
  if len(values) == 0 or separation == 0:
    return (values, list(range(len(values)))) if idxs else values
  selected_values = []
  selected_idxs = []
  threshold = None
  for idx, value in enumerate(values):
    if threshold is None or round(value, precision) >= threshold:
      selected_values.append(value)
      selected_idxs.append(idx)
      threshold = round(value + separation, precision)
  if idxs:
    return selected_values, selected_idxs
  else:
    return selected_values


@T.overload
def select_geq(
  values: T.Sequence[float] | np.ndarray,
  thresholds: T.Sequence[float] | np.ndarray | float,
  *,
  idxs: T.Literal[False] = False,
  precision: int = ...,
) -> list[float]:
  ...


@T.overload
def select_geq(
  values: T.Sequence[float] | np.ndarray,
  thresholds: T.Sequence[float] | np.ndarray | float,
  *,
  idxs: T.Literal[True],
  precision: int = ...,
) -> tuple[list[float], list[int]]:
  ...


def select_geq(
  values: T.Sequence[float] | np.ndarray,
  thresholds: T.Sequence[float] | np.ndarray | float,
  *,
  idxs: bool = False,
  precision: int = 10,
) -> list[float] | tuple[list[float], list[int]]:
  """Selects from a sorted sequence the minimum values that are greater than or equal to each
  threshold (preserving the order of thresholds if multiple are provided)."""
  values = [float(x) for x in values]
  if len(values) == 0:
    return (values, list(range(len(values)))) if idxs else values
  if isinstance(thresholds, (int, float)):
    thresholds = [float(thresholds)]
  else:
    thresholds = [float(x) for x in thresholds]
  if len(thresholds) == 0: return ([], []) if idxs else []
  selected_values = []
  selected_idxs = []
  threshold_idx = 0
  for i, value in enumerate(values):
    if round(value, precision) >= thresholds[threshold_idx]:
      selected_values.append(value)
      selected_idxs.append(i)
      threshold_idx += 1
      if threshold_idx == len(thresholds): break
  if idxs:
    return selected_values, selected_idxs
  else:
    return selected_values
