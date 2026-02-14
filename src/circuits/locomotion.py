import typing as T

import torch

import circuits as cc
import utils.torch as utt


class RhythmGenerationOscKwargs(T.TypedDict, total=False):
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


class RhythmGenerationBasicKwargs(T.TypedDict, total=False):
  voltage_time: float


class RhythmGenerationCircuitInitializer(T.Protocol):
  def __call__(
    self,
    p: utt.containers.ParameterManager,
    grad: bool = False,
  ) -> tuple[RhythmGenerationOscKwargs, RhythmGenerationBasicKwargs]:
    """Initializes parameters and hyperparameters."""
    ...


def activation_locomotion(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
) -> torch.Tensor:
  pos = torch.clip(x, 0, 1)
  return cc.logarithmic(
    v,
    a,
    x,
    active_bound=active_bound,
    quiet_bound=quiet_bound,
    active_start=cc.interpolate_linear(0.85, 0.90, cc.clip_normalized(0.0, 1.0, pos)),
    active_end=cc.interpolate_linear(0.05, 0.2, cc.clip_normalized(0.0, 1.0, pos)),
    quiet=cc.interpolate_linear(0, 0.1, cc.clip_normalized(0.6, 1.0, pos)),
    base=20,
  )


def init_rg_full(p: utt.containers.ParameterManager, grad: bool = False):
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], -0.05, grad)  # flx quiet at baseline
  p.config('bias_ext', ['bias_ext_*'], 1.2, grad)  # ext tonic at baseline
  p.config('bias_v0d', ['bias_v0d_*'], -0., grad)  # active at slow (cross async)
  p.config('bias_v3e', ['bias_v3e_*'], -0., grad)  # inactive at slow (cross sync)
  p.config('bias_v3f', ['bias_v3f_*'], -0.2, grad)  # inactive at slow (cross sync)
  p.config('bias_v0v', ['bias_v0v_*'], -0., grad)  # active at slow (diag async)
  p.config('bias_v3a', ['bias_v3a_*'], -0.7, grad)  # inactive at slow (diag sync)
  p.config('bias_in2', ['bias_in2_*'], -0.5, grad)  # inactive at slow (diag sync)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -2.0, grad)  # flx dominates, osc async
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.1, grad)  # ext has minor influence
  # Cross-side connections (cross).
  p.config('cross_flxflx_i_R', ['cross_flxflx_i_*R'], -1.2, grad)  # cross async at slow, bc inhibit
  p.config('cross_flxflx_i_L', ['cross_flxflx_i_*L'], -1.4, grad)  # promotes async, bc inhibit
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.2, grad)  # cross sync at fast, bc excite
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.2, grad)  # cross sync
  p.config('cross_extflx_i', ['cross_extflx_i_*'], -0.3, grad)  # cross walk, left-right phase 0.5
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -1.2, grad)  # side async
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.05, grad)  # side async
  # Same-side, ascending connections (sidea).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -1.2, grad)  # side async
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.05, grad)  # side async
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -0.8, grad)  # diag async at slow
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.7, grad)  # diag async at slow
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.1, grad)  # diag sync at fast
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1., False)  # identity
  p.config('aff_ext', ['aff_ext_*'], 1., False)  # identity
  # Command connections (freq, sync).
  p.config('freq_flx_e_F', ['freq_flx_e_F*'], 0.65, grad)  # flx rises (2)
  p.config('freq_flx_e_H', ['freq_flx_e_H*'], 0.65, grad)  # flx rises

  p.config('freq_ext_e', ['freq_ext_e_*'], 0., grad)  # ext constant
  p.config('freq_cross_e', ['freq_cross_e_*'], 0.3, grad)  # cross sync at fast (v3f)
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.5, grad)  # diag sync at trot (v3a)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.6, grad)  # cross sync at bound (v0d)
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -1.2, grad)  # diag async walk, sync trot (v0d)
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.6, grad)  # diag sync trot, async bound (v0v)

  # Unit hyperparameters.
  osc: RhythmGenerationOscKwargs = {
    'activation': activation_locomotion,
    'adaptation_time': 450,
    'active_time': 75,
    'quiet_time': 400,
    'active_scale_pos': 0.125,
    'quiet_scale_pos': 0.025,
    'active_scale_neg': 0.4,
    'quiet_scale_neg': 1.1,
    'adaptation_scale_active_pos': 1.,
    'adaptation_scale_quiet_pos': 1.2,
    'adaptation_scale_active_neg': 1.2,
    'adaptation_scale_quiet_neg': 0.8,
    'active_stable': 1,
    'quiet_delay': 0.1,
    'quiet_stable_margin': 50,
    'noise': 0.02,
  }
  basic: RhythmGenerationBasicKwargs = {
    'voltage_time': 10,
  }
  return osc, basic


class RhythmGenerationCircuit(cc.Group):
  """The Rhythm Generation (RG) circuit module coordinates gait between limbs. It is implemented
  using the `circuits` library by connecting Oscillator and Basic units according to genetically
  defined neural circuit models (Danner et al. 2019). The limbs are denoted "FL" (front left), "FR"
  (front right), "HL" (hind left), and "HR" (hind right)."""
  def __init__(
    self,
    init: RhythmGenerationCircuitInitializer = init_rg_full,
    **kwargs,
  ):
    super().__init__(**kwargs)

    # ----------------------------------------------------------------------------------------------
    # Parameters

    self.params = p = utt.containers.ParameterManager()
    p.config('zero', ['*'], 0., False)  # Parameters default to 0, unless initialized otherwise.
    osc_kwargs, basic_kwargs = init(p)

    # ----------------------------------------------------------------------------------------------
    # Circuit units

    # Oscillator units.
    XF = -20
    XH = -70
    XFLX = 0
    YFLX = 21
    XEXT = 0
    YEXT = 24
    # FL
    assert 'adaptation_time' in osc_kwargs
    self.flx_FL = cc.Oscillator(
      bias=p['bias_flx_FL'],
      **{
        **osc_kwargs, 'quiet_delay': 0.07
      },
      xy=(XF - XFLX, +YFLX),
      anchor='y+'
    )
    self.ext_FL = cc.Oscillator(
      bias=p['bias_ext_FL'],
      **osc_kwargs,
      xy=(XF - XEXT, +YEXT),
    )
    # FR
    self.flx_FR = cc.Oscillator(
      bias=p['bias_flx_FR'],
      **{
        **osc_kwargs, 'quiet_delay': 0.02
      },
      xy=(XF - XFLX, -YFLX),
      anchor='y+'
    )
    self.ext_FR = cc.Oscillator(bias=p['bias_ext_FR'], **osc_kwargs, xy=(XF - XEXT, -YEXT))
    # HL
    self.flx_HL = cc.Oscillator(
      bias=p['bias_flx_HL'],
      **{
        **osc_kwargs, 'quiet_delay': 0.07
      },
      xy=(XH - XFLX, +YFLX),
      anchor='y+'
    )
    self.ext_HL = cc.Oscillator(bias=p['bias_ext_HL'], **osc_kwargs, xy=(XH - XEXT, +YEXT))
    # HR
    self.flx_HR = cc.Oscillator(
      bias=p['bias_flx_HR'],
      **{
        **osc_kwargs, 'quiet_delay': 0.02
      },
      xy=(XH - XFLX, -YFLX),
      anchor='y+'
    )
    self.ext_HR = cc.Oscillator(bias=p['bias_ext_HR'], **osc_kwargs, xy=(XH - XEXT, -YEXT))

    # Interneuron units.
    XC_V0D = XFLX
    XC_V3F = XFLX - 5
    XC_V3E = XEXT - 5
    YC_V0D = YFLX - 13
    YC_V3F = YFLX - 11
    YC_V3E = YEXT + 4
    XD_V0D = 12
    XD_V0V = 12
    XD_IN2 = 6
    YD_V0D = 7
    YD_V0V = 11
    YD_IN2 = 5
    XA_V3A = -12
    YA_V3A = 11
    # FL
    self.cross_v0d_FL = cc.Basic(
      bias=p['bias_v0d_FL'], **basic_kwargs, xy=(XF - XC_V0D, +YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_FL = cc.Basic(
      bias=p['bias_v3f_FL'], **basic_kwargs, xy=(XF - XC_V3F, +YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_FL = cc.Basic(
      bias=p['bias_v3e_FL'], **basic_kwargs, xy=(XF - XC_V3E, +YC_V3E), flow='x-'
    )
    self.diagd_v0d_FL = cc.Basic(
      bias=p['bias_v0d_FL'], **basic_kwargs, xy=(XF - XD_V0D, +YD_V0D), anchor='y+'
    )
    self.diagd_v0v_FL = cc.Basic(
      bias=p['bias_v0v_FL'], **basic_kwargs, xy=(XF - XD_V0V, +YD_V0V), anchor='y+'
    )
    self.diagd_in2_FL = cc.Basic(
      bias=p['bias_in2_FL'], **basic_kwargs, xy=(XF - XD_IN2, +YD_IN2), anchor='y+'
    )
    # FR
    self.cross_v0d_FR = cc.Basic(
      bias=p['bias_v0d_FR'], **basic_kwargs, xy=(XF - XC_V0D, -YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_FR = cc.Basic(
      bias=p['bias_v3f_FR'], **basic_kwargs, xy=(XF - XC_V3F, -YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_FR = cc.Basic(
      bias=p['bias_v3e_FR'], **basic_kwargs, xy=(XF - XC_V3E, -YC_V3E), flow='x-'
    )
    self.diagd_v0d_FR = cc.Basic(
      bias=p['bias_v0d_FR'], **basic_kwargs, xy=(XF - XD_V0D, -YD_V0D), anchor='y+'
    )
    self.diagd_v0v_FR = cc.Basic(
      bias=p['bias_v0v_FR'], **basic_kwargs, xy=(XF - XD_V0V, -YD_V0V), anchor='y+'
    )
    self.diagd_in2_FR = cc.Basic(
      bias=p['bias_in2_FR'], **basic_kwargs, xy=(XF - XD_IN2, -YD_IN2), anchor='y+'
    )
    # HL
    self.cross_v0d_HL = cc.Basic(
      bias=p['bias_v0d_HL'], **basic_kwargs, xy=(XH - XC_V0D, +YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_HL = cc.Basic(
      bias=p['bias_v3f_HL'], **basic_kwargs, xy=(XH - XC_V3F, +YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_HL = cc.Basic(
      bias=p['bias_v3e_HL'], **basic_kwargs, xy=(XH - XC_V3E, +YC_V3E), flow='x-'
    )
    self.diaga_v3a_HL = cc.Basic(
      bias=p['bias_v3a_HL'], **basic_kwargs, xy=(XH - XA_V3A, +YA_V3A), flow='x-', anchor='y+'
    )
    # HR
    self.cross_v0d_HR = cc.Basic(
      bias=p['bias_v0d_HR'], **basic_kwargs, xy=(XH - XC_V0D, -YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_HR = cc.Basic(
      bias=p['bias_v3f_HR'], **basic_kwargs, xy=(XH - XC_V3F, -YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_HR = cc.Basic(
      bias=p['bias_v3e_HR'], **basic_kwargs, xy=(XH - XC_V3E, -YC_V3E), flow='x-'
    )
    self.diaga_v3a_HR = cc.Basic(
      bias=p['bias_v3a_HR'], **basic_kwargs, xy=(XH - XA_V3A, -YA_V3A), flow='x-', anchor='y+'
    )

    # Afferent units.
    XAF_FLX = XFLX - 5
    YAF_FLX = YFLX - 6
    YAF_FLXH = YFLX - 8
    XAF_EXT = XEXT - 5
    YAF_EXT = YEXT + 2
    # FL
    self.aff_flx_FL = cc.Signal(xy=(XF - XAF_FLX, +YAF_FLX), anchor='y+')
    self.aff_ext_FL = cc.Signal(xy=(XF - XAF_EXT, +YAF_EXT))
    # FR
    self.aff_flx_FR = cc.Signal(xy=(XF - XAF_FLX, -YAF_FLX), anchor='y+')
    self.aff_ext_FR = cc.Signal(xy=(XF - XAF_EXT, -YAF_EXT))
    # HL
    self.aff_flx_HL = cc.Signal(xy=(XH - XAF_FLX, +YAF_FLXH), anchor='y+')
    self.aff_ext_HL = cc.Signal(xy=(XH - XAF_EXT, +YAF_EXT))
    # HR
    self.aff_flx_HR = cc.Signal(xy=(XH - XAF_FLX, -YAF_FLXH), anchor='y+')
    self.aff_ext_HR = cc.Signal(xy=(XH - XAF_EXT, -YAF_EXT))

    # Command units.
    XCMD = -1
    YFREQ = 3
    YSYNC = 1
    # R
    self.freq_R = cc.Signal(xy=(XCMD, -YFREQ), anchor='y+')
    self.sync_R = cc.Signal(xy=(XCMD, -YSYNC))
    # L
    self.freq_L = cc.Signal(xy=(XCMD, +YFREQ), anchor='y+')
    self.sync_L = cc.Signal(xy=(XCMD, +YSYNC))

    # ----------------------------------------------------------------------------------------------
    # Circuit connections

    # Alias self to ensure that named access of circuit units typechecks correctly.
    u = T.cast(T.Mapping[str, cc.Basic | cc.Oscillator], self)

    # Oscillator connections (osc).
    for a in ('FL', 'FR', 'HL', 'HR'):
      u[f'flx_{a}'].synapse(u[f'ext_{a}'], weight=p[f'osc_extflx_i_{a}'], sign='-')
      u[f'ext_{a}'].synapse(u[f'flx_{a}'], weight=p[f'osc_flxext_i_{a}'], sign='-')

    # Cross-side connections (cross).
    for a, b in (('FR', 'FL'), ('FL', 'FR'), ('HR', 'HL'), ('HL', 'HR')):
      u[f'cross_v3f_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v3e_{a}'].synapse(u[f'ext_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(u[f'cross_v0d_{a}'], weight=p[f'cross_flxflx_i_{b}'], sign='-')
      u[f'flx_{b}'].synapse(
        u[f'cross_v3e_{a}'], weight=p[f'cross_extflx_i_{b}'], sign='-', waypoints=[(-8, +4, 'y-')]
      )
      u[f'flx_{b}'].synapse(u[f'cross_v3f_{a}'], weight=p[f'cross_flxflx_e_{b}'], sign='+')
      u[f'ext_{b}'].synapse(
        u[f'cross_v3e_{a}'], weight=p[f'cross_extext_e_{b}'], sign='+', waypoints=[(-8, -1, 'y-')]
      )

    # Same-side, descending connections (sided).
    for a, b in (('FL', 'HL'), ('FR', 'HR')):
      u[f'flx_{b}'].synapse(
        u[f'ext_{a}'], weight=p[f'sided_extflx_e_{b}'], sign='+', waypoints=[(-10, -2, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'flx_{a}'], weight=p[f'sided_flxflx_i_{b}'], sign='-', waypoints=[(-11, +2, 'y+')]
      )

    # Same-side, ascending connections (sidea).
    for a, b in (('HL', 'FL'), ('HR', 'FR')):
      u[f'flx_{b}'].synapse(
        u[f'ext_{a}'], weight=p[f'sidea_extflx_e_{b}'], sign='+', waypoints=[(-3, -7, 'y-')]
      )

    # Diagonal, descending connections (diagd).
    for a, b in (('FL', 'HR'), ('FR', 'HL')):
      u[f'diagd_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'diagd_v0v_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0d_{a}'],
        weight=p[f'diagd_flxflx_i_{b}'],
        sign='-',
        waypoints=[(-35, +12, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0v_{a}'],
        weight=p[f'diagd_flxflx_e_{b}'],
        sign='+',
        waypoints=[(-34, +12, 'y-')]
      )
      u[f'diagd_in2_{a}'].synapse(u[f'cross_v3f_{a}'], weight=1, sign='+')
      u[f'diagd_v0d_{a}'].synapse(u[f'diagd_in2_{a}'], weight=p[f'diagd_in2v0d_i_{a}'], sign='-')

    # Diagonal, ascending connections (diaga).
    for a, b in (('HL', 'FR'), ('HR', 'FL')):
      u[f'diaga_v3a_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diaga_v3a_{a}'],
        weight=p[f'diaga_flxflx_e_{b}'],
        sign='+',
        waypoints=[(32, 8.5, 'x-'), (-3, 7, 'y-')]
      )
      u[f'diagd_in2_{b}'].synapse(
        u[f'diaga_v3a_{a}'],
        weight=p[f'diaga_v3ain2_e_{b}'],
        sign='+',
        waypoints=[(26, -7.5, 'x-')]
      )

    # Afferent connections (aff).
    for a in ('FL', 'FR', 'HL', 'HR'):
      u[f'flx_{a}'].synapse(u[f'aff_flx_{a}'], weight=p[f'aff_flx_{a}'])
      u[f'ext_{a}'].synapse(u[f'aff_ext_{a}'], weight=p[f'aff_ext_{a}'])

    # Command connections (freq, sync).
    for a in ('L', 'R'):
      u[f'flx_F{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_flx_e_F{a}'], sign='+', waypoints=[(-10, 12, 'y-')]
      )
      u[f'flx_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_flx_e_H{a}'], sign='+', waypoints=[(-10, 12, 'y-')]
      )
      u[f'ext_F{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_ext_e_F{a}'], sign='+', waypoints=[(-10, -15, 'y-')]
      )
      u[f'ext_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_ext_e_H{a}'], sign='+', waypoints=[(-9, -15, 'y-')]
      )
      u[f'cross_v3f_F{a}'].synapse(u[f'freq_{a}'], weight=p[f'freq_cross_e_{a}'], sign='+')
      u[f'cross_v3f_H{a}'].synapse(u[f'freq_{a}'], weight=p[f'freq_cross_e_{a}'], sign='+')
      u[f'diaga_v3a_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_diaga_e_{a}'], sign='+', waypoints=[(-2, 3, 'y-')]
      )
      u[f'cross_v0d_F{a}'].synapse(u[f'sync_{a}'], weight=p[f'sync_cross_i_{a}'], sign='-')
      u[f'cross_v0d_H{a}'].synapse(u[f'sync_{a}'], weight=p[f'sync_cross_i_{a}'], sign='-')
      u[f'diagd_v0d_F{a}'].synapse(
        u[f'sync_{a}'], weight=p[f'sync_diagdi_i_{a}'], sign='-', waypoints=[(-2, 3.5, 'y-')]
      )
      u[f'diagd_v0v_F{a}'].synapse(
        u[f'sync_{a}'], weight=p[f'sync_diagde_i_{a}'], sign='-', waypoints=[(-2, 3, 'y-')]
      )
