import abc

import numpy as np
import numpy.typing as npt


class CartesianModel(abc.ABC):
  @abc.abstractmethod
  def step(
    self,
    v: np.ndarray,
    a: np.ndarray,
    i: np.ndarray,
    dt: float,
  ) -> tuple[np.ndarray, np.ndarray]:
    pass

  @abc.abstractmethod
  def dv_da(
    self,
    v: np.ndarray,
    a: np.ndarray,
    i: np.ndarray | float = 0.,
  ) -> tuple[np.ndarray, np.ndarray]:
    pass

  @abc.abstractmethod
  def y(
    self,
    v: np.ndarray,
    a: np.ndarray,
    i: np.ndarray | float = 0.,
  ) -> np.ndarray:
    pass


class CartesianModelWithNullclines(CartesianModel):
  @abc.abstractmethod
  def v_nullcline(
    self,
    v: np.ndarray,
    a: np.ndarray,
    i: np.ndarray | float = 0.,
  ) -> tuple[np.ndarray, np.ndarray] | None:
    pass

  @abc.abstractmethod
  def a_nullcline(
    self,
    v: np.ndarray,
    a: np.ndarray,
    i: np.ndarray | float = 0.,
  ) -> tuple[np.ndarray, np.ndarray] | None:
    pass


class PolarModel:
  @abc.abstractmethod
  def step(
    self,
    r: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    dt: float,
  ) -> tuple[np.ndarray, np.ndarray]:
    pass

  @abc.abstractmethod
  def dv_da(
    self,
    r: np.ndarray,
    p: np.ndarray,
    i: np.ndarray | float = 0.,
  ) -> tuple[np.ndarray, np.ndarray]:
    pass

  @abc.abstractmethod
  def y(
    self,
    r: np.ndarray,
    p: np.ndarray,
    i: np.ndarray | float = 0.,
  ) -> np.ndarray:
    pass

  @abc.abstractmethod
  def v(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    pass

  @abc.abstractmethod
  def a(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    pass
