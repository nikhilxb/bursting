"""
This module defines `SimulatorConfig`, which holds configuration settings for the simulator.
"""
import dataclasses


@dataclasses.dataclass
class SimulatorConfig:
  timestep: float = 10
  """Time (in milliseconds) represented by a single step of the simulation."""
