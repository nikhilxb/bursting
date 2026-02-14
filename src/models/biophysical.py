import typing as T

import numpy as np
import numpy.typing as npt

from . import base


class BiophysicalOscillator(base.CartesianModelWithNullclines):
  def __init__(
    self,
    C: float = 20.,  # Capacitance [pF], a.k.a voltage time constant.
    G_leak: float = 4.5,  # Leakage conductance [nS].
    V_leak: float = -62.5,  # Leakage reversal potential [mV].
    G_exc: float = 10.,  # Excitatory conductance [nS].
    V_exc: float = -10.,  # Excitatory reversal potential [mV].
    D_exc: float = 0.02,  # Excitatory drive.
    G_inh: float = 10.,  # Inhibitory conductance [nS].
    V_inh: float = -75.0,  # Inhibitory reversal potential [mV].
    D_inh: float = 0.,  # Inhibitory drive.
    G_nap: float = 4.5,  # Sodium conductance [nS].
    V_nap: float = 55.,  # Sodium reversal potential [mV].
    V_m: float = -40.,  # Sodium activation midpoint [mV].
    K_m: float = 6.,  # Sodium activation slope.
    V_a: float = -45.,  # Sodium inactivation midpoint [mV].
    K_a: float = -4.,  # Sodium inactivation slope.
    V_ta: float = -35.,  # Sodium inactivation time constant midpoint [mV].
    K_ta: float = 15.,  # Sodium inactivation time constant scale.
    Ta: float = 320.,  # Sodium inactivation time constant baseline [ms]
    Ta_max: float = 640.,  # Sodium inactivation time constant maximum [ms].
    V_threshold: float = -50.,  # Firing threshold voltage [mV].
    V_max: float = 0.,  # Firing maximum voltage [mV].
  ):
    super().__init__()
    assert C > 0
    assert Ta > 0
    self.C = C
    self.G_leak = G_leak
    self.V_leak = V_leak
    self.G_exc = G_exc
    self.V_exc = V_exc
    self.D_exc = D_exc
    self.G_inh = G_inh
    self.V_inh = V_inh
    self.D_inh = D_inh
    self.G_nap = G_nap
    self.V_nap = V_nap
    self.V_m = V_m
    self.K_m = K_m
    self.V_a = V_a
    self.K_a = K_a
    self.V_ta = V_ta
    self.K_ta = K_ta
    self.Ta = Ta
    self.Ta_max = Ta_max
    self.V_threshold = V_threshold
    self.V_max = V_max

  def step(self, v, a, i, dt):
    # Forward-Euler discretization.
    dv, da = self.dv_da(v, a, i)
    v = v + dv * dt
    a = a + da * dt
    return v, a

  def dv_da(self, v, a, i=0):
    # Voltage dynamics.
    dv = (-self._i_exc(v, i) - self._i_inh(v, i) - self._i_nap(v, a) - self._i_leak(v)) / self.C
    # Activation dynamics.
    da = (self._nap_a_inf(v) - a) / self._nap_a_time(v)
    return dv, da

  def _i_leak(self, v: np.ndarray):
    # Leakage current.
    return self.G_leak * (v - self.V_leak)

  def _i_exc(self, v: np.ndarray, i: np.ndarray | float) -> np.ndarray:
    # Synaptic current (excitatory).
    return self.G_exc * (v - self.V_exc) * (self.D_exc + np.clip(i, 0, None))

  def _i_inh(self, v: np.ndarray, i: np.ndarray | float) -> np.ndarray:
    # Synaptic current (inhibitory).
    return self.G_inh * (v - self.V_inh) * (self.D_inh - np.clip(i, None, 0))

  def _i_nap(self, v: np.ndarray, a: np.ndarray):
    # Persistent sodium current.
    return self._nap_z(v) * a

  def _nap_z(self, v: np.ndarray):
    # Persistent sodium current maximum.
    return self.G_nap * self._nap_m_inf(v) * (v - self.V_nap)

  def _nap_m_inf(self, v: np.ndarray):
    # Persistent sodium channel activation.
    return 1 / (1 + np.exp(-(v - self.V_m) / self.K_m))  # fraction [0, 1]

  def _nap_a_inf(self, v: np.ndarray):
    # Adaptation steady-state, a.k.a. persistent sodium channel inactivation.
    return 1 / (1 + np.exp(-(v - self.V_a) / self.K_a))  # fraction [0, 1]

  def _nap_a_time(self, v: np.ndarray):
    # Adaptation time constant, ak.a. persistent sodium channel inactivation time constant.
    return self.Ta + (self.Ta_max - self.Ta) / np.cosh((v - self.V_ta) / self.K_ta)

  def y(self, v, a, i=0):
    # Firing rate.
    return np.clip((v - self.V_threshold) / (self.V_max - self.V_threshold), 0, 1)

  def v_nullcline(self, v, a, i=0):
    return v, (-self._i_exc(v, i) - self._i_inh(v, i) - self._i_leak(v)) / self._nap_z(v)

  def a_nullcline(self, v, a, i=0):
    return v, self._nap_a_inf(v)
