import typing as T

import torch

import circuits as cc
import utils.torch as utt


class PyloricOscKwargs(T.TypedDict, total=False):
  activation: T.Callable
  adaptation_time: float
  active_time: float
  quiet_time: float
  active_scale_pos: float
  quiet_scale_pos: float
  active_scale_neg: float
  quiet_scale_neg: float
  adaptation_scale_active_pos: float
  adaptation_scale_quiet_pos: float
  adaptation_scale_active_neg: float
  adaptation_scale_quiet_neg: float
  active_stable: float
  quiet_stable: float
  active_stable_margin: float
  quiet_stable_margin: float
  active_delay: float
  quiet_delay: float
  noise: float
  bias: float


class PyloricCircuitInitializer(T.Protocol):
  def __call__(
    self,
    p: utt.containers.ParameterManager,
    grad: bool = False,
    scale: float = 1.,
  ) -> tuple[PyloricOscKwargs, PyloricOscKwargs, PyloricOscKwargs]:
    """Initializes parameters and hyperparameters."""
    ...


def activation_pyloric(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
) -> torch.Tensor:
  return (v >= 0) * cc.interpolate_linear(0.5, 1.0, a)


def init_pyloric(p: utt.containers.ParameterManager, grad: bool = False, scale: float = 1.):
  # Synaptic connections.
  p.config('PD_LP_i', 'PD_LP_i', -1.0, grad)
  p.config('PD_PY_i', 'PD_PY_i', -1.0, grad)
  p.config('LP_PD_i', 'LP_PD_i', -0.1, grad)
  p.config('LP_PY_i', 'LP_PY_i', -0.5, grad)
  p.config('PY_LP_i', 'PY_LP_i', -0.5, grad)

  # Unit hyperparamters.
  PD: PyloricOscKwargs = {
    'activation': activation_pyloric,
    'adaptation_time': 1500 * scale,
    'active_time': 200 * scale,
    'quiet_time': 800 * scale,
    'active_scale_pos': 1,
    'quiet_scale_pos': 1,
    'active_scale_neg': 0.9,
    'quiet_scale_neg': 1.1,
    'bias': 0.,
    'noise': 0.05,
  }
  LP: PyloricOscKwargs = {
    'activation': activation_pyloric,
    'adaptation_time': 1500 * scale,
    'active_time': 250 * scale,
    'quiet_time': 750 * scale,
    'active_scale_pos': 1,
    'quiet_scale_pos': 1,
    'active_scale_neg': 0.5,
    'quiet_scale_neg': 1.3,
    'adaptation_scale_active_pos': 1.,
    'adaptation_scale_quiet_pos': 1.1,
    'adaptation_scale_active_neg': 1.1,
    'adaptation_scale_quiet_neg': 0.5,
    'active_delay': 0,
    'quiet_delay': 0.25,
    'quiet_stable_margin': 10 * scale,
    'bias': -0.1,
    'noise': 0.05,
  }
  PY: PyloricOscKwargs = {
    'activation': activation_pyloric,
    'adaptation_time': 1500 * scale,
    'active_time': 250 * scale,
    'quiet_time': 750 * scale,
    'active_scale_pos': 1,
    'quiet_scale_pos': 1,
    'active_scale_neg': 0.5,
    'quiet_scale_neg': 1.3,
    'adaptation_scale_active_pos': 1.,
    'adaptation_scale_quiet_pos': 1.1,
    'adaptation_scale_active_neg': 1.1,
    'adaptation_scale_quiet_neg': 0.5,
    'active_delay': 0,
    'quiet_delay': 0.95,
    'quiet_stable_margin': 10 * scale,
    'bias': -0.1,
    'noise': 0.05,
  }
  return PD, LP, PY


class PyloricCircuit(cc.Group):
  def __init__(
    self,
    synapses: bool = True,
    scale: float = 1.,
    init: PyloricCircuitInitializer = init_pyloric,
    **kwargs,
  ):
    super().__init__(**kwargs)

    self.params = p = utt.containers.ParameterManager()
    p.config('zero', ['*'], 0., False)  # Parameters default to 0, unless initialized otherwise.
    PD_kwargs, LP_kwargs, PY_kwargs = init(p, scale=scale)

    # Circuit units.
    XPD = 5
    YPD = 10
    XLP = 15
    YLP = 14
    XPY = 15
    YPY = 5
    self.PD = PD = cc.Oscillator(**PD_kwargs, xy=(XPD, YPD), anchor='y+', flow='x+', decorator='~')
    self.LP = LP = cc.Oscillator(**LP_kwargs, xy=(XLP, YLP), anchor='y+', flow='x+')
    self.PY = PY = cc.Oscillator(**PY_kwargs, xy=(XPY, YPY), anchor='y-', flow='x+')

    # Circuit connections.
    if synapses:
      LP.synapse(PD, weight=p['PD_LP_i'], sign='-')
      PY.synapse(PD, weight=p['PD_PY_i'], sign='-')
      PD.synapse(LP, weight=p['LP_PD_i'], sign='-', waypoints=[(0, -5, 'x-')])
      PY.synapse(LP, weight=p['LP_PY_i'], sign='-')
      LP.synapse(PY, weight=p['PY_LP_i'], sign='-')
