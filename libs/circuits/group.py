"""
This module defines the `Group` class, which is a container node that groups nodes together.
"""

from circuits import nodes


class Group(nodes.ContainerNode):
  """A container node that groups nodes together. It mainly is used for organizing nodes in a
  modular, hierarchical way. Children may inherit the parent's metadata (e.g. rendering options,
  layout relative to the parent's layout coordinates).
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
