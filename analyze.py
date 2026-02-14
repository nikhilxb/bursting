import dataclasses
import logging
import os
import pathlib
import time
import typing as T

import hydra
import pandas as pd
import ray

import infra.launcher  # Import to register plugins.
import utils.io

from src import models

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PathsConfigNode:
  output_dir: str
  working_dir: str
  checkpoint_dir: str | None


@dataclasses.dataclass
class AnalyzeConfig:
  model: models.CartesianModel | models.PolarModel
  simulate: T.Callable[..., pd.DataFrame]
  measure: T.Callable | None
  plot: T.Callable | None
  paths: PathsConfigNode


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: AnalyzeConfig) -> None:
  # Setup compute and output directory.
  ray.init(num_cpus=len(os.sched_getaffinity(0)), ignore_reinit_error=True)
  output_dir = pathlib.Path(cfg.paths.output_dir)
  os.makedirs(output_dir, exist_ok=True)
  log.info(f'Initialized ray with resources: {ray.available_resources()}')
  log.info(f'Initialized output directory: {output_dir}')

  # Instantiate simulation functions.
  model = hydra.utils.instantiate(cfg.model)
  simulate = hydra.utils.call(cfg.simulate, model, _partial_=True)

  # Measure data, or load from file.
  if cfg.paths.checkpoint_dir is None:
    assert cfg.measure is not None
    log.info(f'Started measure.')
    start_time = time.time()
    data = hydra.utils.call(cfg.measure, simulate)
    utils.io.write_file(output_dir / 'data.pkl', data)
    measure_time = time.time() - start_time
    log.info(f'Finished measure in {measure_time:.2f} seconds.')
  else:
    checkpoint_path = pathlib.Path(cfg.paths.checkpoint_dir) / 'data.pkl'
    data = utils.io.read_file(checkpoint_path)
    measure_time = None
    log.info(f'Skipped measure. Loaded checkpoint: {checkpoint_path}')

  # Plot data, or skip.
  if cfg.plot:
    log.info(f'Started plot.')
    start_time = time.time()
    hydra.utils.call(
      cfg.plot,
      data,
      output_dir=output_dir,
      x_nullcline=getattr(model, 'v_nullcline', None),
      y_nullcline=getattr(model, 'a_nullcline', None),
    )
    plot_time = time.time() - start_time
    log.info(f'Finished plot in {plot_time:.2f} seconds.')
  else:
    plot_time = None
    log.info(f'Skipped plot.')

  # Write jobs stats.
  utils.io.write_file(
    output_dir / 'stats.yaml',
    dict(
      measure_time=measure_time,
      plot_time=plot_time,
    ),
  )


if __name__ == '__main__':
  main()
