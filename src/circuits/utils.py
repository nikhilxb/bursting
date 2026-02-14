import os
import pathlib

import mediapy
import numpy as np
import pandas as pd
import torch

import circuits as cc
import circuits.rendering as ccr

from .. import stimuli


def _getnestedattr(obj, attr: str) -> torch.Tensor:
  for a in attr.split('.'):
    obj = getattr(obj, a)
  return obj


def simulate(
  sim: cc.Simulator,
  T: float = 1000,
  seed: int = 0,
  info: bool = False,
  image: bool = False,
  video: bool = False,
  fps: float = 30,
  dpi: int = 100,
  inputs: dict[str, stimuli.Stimulus] = {},
  outputs: list[str] = [],
  show: bool = True,
  save: bool = False,
  save_dir: str | pathlib.Path | os.PathLike = 'outputs/neurons',
  **kwargs,
) -> pd.DataFrame:
  save_dir = pathlib.Path(save_dir)
  if save: os.makedirs(save_dir, exist_ok=True)

  torch.manual_seed(seed)
  np.random.seed(seed)

  renderer = ccr.Renderer(sim, dpi=dpi, **kwargs)
  sim.init()
  if info: print(sim)

  frames = []
  if image:
    if save: renderer.render_to_file(save_dir / 'circuit.pdf')
    if show: mediapy.show_image(renderer.render()[:, :, :3])
  if video:
    frames.append(renderer.render()[:, :, :3])  # Ignore alpha channel for videos.

  def inject(t: float, T: float):
    values = []
    for name, stimulus in inputs.items():
      value = stimulus(t, T)
      _getnestedattr(sim, name).copy_(torch.as_tensor(value))
      values.append(value)
    return values

  def extract():
    values = []
    for name in outputs:
      try:
        value = _getnestedattr(sim, name).item()
      except AttributeError:
        value = None
      values.append(value)
    return values

  data = []
  dt_render = 1000 / fps
  t_render = 0.
  while (t := sim.time) <= T:
    data.append((t, *inject(t, T), *extract()))
    sim.step()
    if video and (t >= t_render or t + sim.cfg.timestep >= T):
      frames.append(renderer.render()[:, :, :3])
      t_render += dt_render

  if video:
    if save: mediapy.write_video(save_dir / 'circuit.mp4', frames, fps=fps)
    if show: mediapy.show_video(frames, fps=fps, loop=False)

  return pd.DataFrame(data, columns=['t', *inputs.keys(), *outputs])
