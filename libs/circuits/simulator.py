"""
This module defines the `Simulator` class, which manages the computations of a neural circuit tree.
"""
import functools

from circuits import config, generators, monitors, nodes, units


def _simulator_init(node: nodes.ModuleNode, cfg: config.SimulatorConfig):
  if isinstance(node, (generators.Generator, units.Unit, monitors.Monitor)):
    node.init(cfg)


def _simulator_reset(node: nodes.ModuleNode):
  if isinstance(node, (generators.Generator, units.Unit, monitors.Monitor)):
    node.reset()


def _simulator_step(node: nodes.ModuleNode):
  # Update values, no dependencies.
  if isinstance(node, (generators.Generator, units.Unit)):
    node.step()


def _simulator_populate(node: nodes.ModuleNode):
  # Read values, via dependencies.
  if isinstance(node, (units.Unit, monitors.Monitor)):
    node.populate()


class Simulator(nodes.ContainerNode):
  """The top-level node that manages the computations of a neural circuit tree. It recursively
  initializes, resets, and updates nodes."""
  def __init__(self, **kwargs):
    super().__init__()
    self.cfg = config.SimulatorConfig(**kwargs)
    assert self.cfg.timestep > 0

    self.steps: int = 0
    """Number of elapsed simulated steps."""

    self.time: float = 0.
    """Duration of elapsed simulated time (in miliseconds), which depends on `cfg.timestep`."""

  def init(self):
    """Initializes the simulator and resets it to its initial state."""
    nodes.visit(self, functools.partial(_simulator_init, cfg=self.cfg))
    self.reset()

  def reset(self):
    """Resets the simulator to its initial state."""
    nodes.visit(self, _simulator_reset)
    self.steps = 0
    self.time = 0.

  def step(self):
    """Advances the simulator to an updated state."""
    # Some nodes may have dependencies on others, so we need to traverse in the right order.
    nodes.visit(self, _simulator_step)
    nodes.visit(self, _simulator_populate)
    self.steps += 1
    self.time += self.cfg.timestep
