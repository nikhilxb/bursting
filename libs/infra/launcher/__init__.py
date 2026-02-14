import functools
import math
import os
import pathlib
import re
import shlex
import signal

import omegaconf as oc
import submitit
from hydra.core import config_search_path, plugins
from hydra.experimental import callback
from hydra.plugins import search_path_plugin

import utils.string

# ==================================================================================================
# Configs

# Absolute path to infrastructure configs directory.
CONFIGS_PATH = (pathlib.Path(__file__).parent / 'configs').resolve()


# Extend Hydra config search path to include infrastructure configs. See
# https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin.
class InfraSearchPathPlugin(search_path_plugin.SearchPathPlugin):
  def manipulate_search_path(self, search_path: config_search_path.ConfigSearchPath) -> None:
    search_path.append(provider='infra', path=f'file://{CONFIGS_PATH}')


plugins.Plugins.instance().register(InfraSearchPathPlugin)

# ==================================================================================================
# Logs


class SubmititLogsCallback(callback.Callback):
  """Callback to symlink Submitit logs to the Hydra job directories."""
  def on_job_start(self, config: oc.DictConfig, **kwargs) -> None:
    job_dir = pathlib.Path(config.hydra.runtime.output_dir)
    job_id = str(config.hydra.job.id)
    submitit_dir = pathlib.Path('../.submitit')  # Use relative path to enable moving sweep dir.
    out_path = submitit_dir / f'{job_id}/{job_id}_0_log.out'
    err_path = submitit_dir / f'{job_id}/{job_id}_0_log.err'
    if (job_dir / out_path).is_file(): (job_dir / 'stdout.log').symlink_to(out_path)
    if (job_dir / err_path).is_file(): (job_dir / 'stderr.log').symlink_to(err_path)


# ==================================================================================================
# Containers

# Absolute path to infrastructure containers directory.
CONTAINERS_PATH = (pathlib.Path(__file__).parent / '../containers').resolve()


class RunInContainerCallback(callback.Callback):
  """Callback to run Python within a container. The container script must be a drop-in replacement
  for the `python [args]` command from the terminal, which spins up a container and run the Python
  interpreter with the specified arguments."""
  def __init__(self, container: str | None = None):
    """
    Args:
      container: Path to the container. If `None`, no container is used. If a relative path, it is
        resolved relative to `infra/containers`. If an absolute path, it is used as is. The path may
        include terminal expansions (e.g. `$USER`, `~/path/to/container`).
    Raises:
      ValueError: If the container is not found.
    """
    if not container:
      self.container = None
    else:
      path = pathlib.Path(os.path.expandvars(os.path.expanduser(container)))
      self.container = path.resolve() if path.is_absolute() else (CONTAINERS_PATH / path).resolve()
      if not self.container.is_file():
        raise ValueError(f"Container not found: {self.container}")

  def on_multirun_start(self, config, **kwargs) -> None:
    self._patch_submitit()

  def on_run_start(self, config, **kwargs) -> None:
    self._patch_submitit()

  def _patch_submitit(self) -> None:
    if not self.container: return

    # Monkey-patch Submitit to run Python within the container.
    # https://github.com/facebookincubator/submitit/blob/1.4.5/submitit/slurm/slurm.py#L341-L345
    container = str(self.container)

    def _submitit_command_str(self) -> str:
      return f'{container} -u -m submitit.core._submit {shlex.quote(str(self.folder))}'

    submitit.SlurmExecutor._submitit_command_str = property(_submitit_command_str)  # type: ignore


# ==================================================================================================
# Resolvers

# Register resolvers for convenience when writing configs.
oc.OmegaConf.register_new_resolver('sum', lambda *xs: sum(xs))
oc.OmegaConf.register_new_resolver('prod', lambda *xs: math.prod(xs))
oc.OmegaConf.register_new_resolver('or', lambda *xs: functools.reduce(lambda a, b: a or b, xs))
oc.OmegaConf.register_new_resolver('and', lambda *xs: functools.reduce(lambda a, b: a and b, xs))
oc.OmegaConf.register_new_resolver('replace', lambda s, a, b: s.replace(a, b))
oc.OmegaConf.register_new_resolver('lower', lambda s: s.lower())
oc.OmegaConf.register_new_resolver('upper', lambda s: s.upper())

# Register resolver to convert `override_dirname` to shorter and path-safe directory name, without
# slashes and only including the retargeted option key/value (e.g. "/group1/group2@foo=option1" ->
# "foo=option1").
pattern = re.compile(r"([a-zA-Z0-9._-]+@)?([a-zA-Z0-9._-]+=)")
oc.OmegaConf.register_new_resolver("overrides", lambda s: pattern.sub(r"\2", s.replace('/', '.')))

# Register resolver to create a human-readable unique ID.
oc.OmegaConf.register_new_resolver('hruid', utils.string.hruid, use_cache=True)


# Register resolver to create an experiment ID from the given name or config name.
def _exp_id(name: str | None, config_name: str) -> str:
  exp_id = name or config_name
  exp_id = exp_id.removesuffix('.yaml').removesuffix('.yml')
  return exp_id


oc.OmegaConf.register_new_resolver('exp_id', _exp_id, use_cache=True)


# Register resolver to create a sweep ID from the timestamp, hruid, and tag.
def _sweep_id(now: str, hruid: str, tag: str | None) -> str:
  sweep_id = f"{now}-{tag + ':' if tag else ''}{hruid}"
  return sweep_id


oc.OmegaConf.register_new_resolver('sweep_id', _sweep_id, use_cache=True)

# ==================================================================================================
# Signals

# Override SIGTERM handler so spawned processes don't adopt Submitit's bypass behavior.
# This code will need to be run within the Hydra entrypoint function as well.
# See: https://github.com/facebookincubator/submitit/issues/1677
signal.signal(signal.SIGTERM, lambda *args, **kwargs: exit(0))
