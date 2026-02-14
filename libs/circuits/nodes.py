"""
This module defines the `Node` interfaces/superclasses that compose a circuit.

An architecture built with this circuits library is composed of a tree of `ModuleNode` instances,
which are `torch.nn.Module`/`torch.nn.ModuleDict` with additional methods and metadata. The
top-level node should be a `Simulator` node, which manages the computations of the circuit tree.
Only `ModuleNode` instances (not regular `torch.nn.Module`) may be used as intermediate nodes in
the circuit tree, but `PrimitiveNode` leaves may have additional `torch.nn.Module` as children.
"""
import itertools
import typing as T

import torch

from circuits import config
from utils import collections


class Node:
  """Base class for nodes in the neural circuit.

  Attributes:
    id (int): Unique identifier of the node.
    meta (collections.AttrDict): Metadata of the node, e.g. for rendering settings.
  """

  # Generator for unique `Node` IDs that are deterministic based on `Node` instatiation order.
  _newid = itertools.count()

  def __init__(self, **kwargs):
    self.id: int = next(Node._newid)
    self.meta: collections.AttrDict = collections.AttrDict(kwargs)
    super().__init__()  # Required for multiple-inheritance MRO to `torch.nn.Module`.

  def init(self, cfg: config.SimulatorConfig):
    """Initializes the node with the given simulator configuration.

    Args:
      cfg: Simulator configuration.
    """
    pass

  def reset(self):
    """Resets the node to its initial state."""
    pass


class PrimitiveNode(Node, torch.nn.Module):
  """A circuit tree (leaf) node that cannot contain other nodes."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)


class ContainerNode(Node, torch.nn.ModuleDict):
  """A circuit tree (intermediate) node that can contain other nodes."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)


ModuleNode = PrimitiveNode | ContainerNode
"""Nodes that are `torch.nn.Module`/`torch.nn.ModuleDict` instances."""


def visit(
  root: ModuleNode,
  visitor: T.Callable[[ModuleNode], bool | None],
):
  """Traverses a `Node` tree , applying a visitor function to each node.

  The visitor function can control whether to continue traversing the subtree of a node.

  Args:
    root: Root node of the tree to traverse.
    visitor: Function to apply to each node. If the function returns `False` at a node, the
      subtree rooted at that node will not be traversed.
  """
  def traverse(node: ModuleNode):
    traverse_subtree = visitor(node)
    if traverse_subtree == False or isinstance(node, PrimitiveNode): return
    for child in node.children():
      # Force unbroken subtree of `Node`s, only allowing regular `Module`s as leaves.
      if isinstance(child, (PrimitiveNode, ContainerNode)):
        traverse(child)

  traverse(root)
