"""
This module defines the `Path` classes, which are drawables that represent connections between
circuit elements.
"""
import enum
import typing as T

import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

from circuits.rendering import colors, renderer

COLORS = colors.COLORS


class Side(enum.IntEnum):
  """Represents the 4 sides of a grid square."""
  EAST = 0
  SOUTH = 1
  WEST = 2
  NORTH = 3


class Direction(enum.IntEnum):
  """Represents the 4 directions in which a path can point."""
  # This order is important for rotation calculations, in which directions are converted into a
  # normalized frame defined by a reference direction of `Direction.RIGHT`.
  RIGHT = 0
  DOWN = 1
  LEFT = 2
  UP = 3


PointAndDirection = tuple[float, float, Direction]
"""Tuple `(x, y, direction)` used to specify a path point and direction through it."""


def _dir_rotate(dir: Direction, ref: Direction) -> Direction:
  """Calculates the direction that `dir` corresponds to when `ref` is rotated to become
  `Direction.RIGHT` (i.e. the reference direction of a normalized frame).

  For example, `Direction.UP` in the global frame corresponds to `Direction.LEFT` when the reference
  direction is `Direction.DOWN`.

  Args:
    dir: Query direction.
    ref: Reference direction.
  """
  return list(Direction)[dir - ref]


def _box_startpoint(
  anchor: tuple[int, int],
  side: Side,
  size: tuple[int, int] = (1, 1),
) -> PointAndDirection:
  """Calculates the starting point/direction of a path starting from the given side of the given
  grid bounding box.

  Args:
    anchor: Grid box's anchor point (lower-left).
    side: Side from which the path emerges.
    size: Grid box's size (width, height).
  """
  assert len(anchor) == 2
  x, y = anchor
  assert len(size) == 2
  w, h = size
  match side:
    case Side.EAST:
      return (x + w, y + h / 2, Direction.RIGHT)
    case Side.SOUTH:
      return (x + w / 2, y, Direction.DOWN)
    case Side.WEST:
      return (x, y + h / 2, Direction.LEFT)
    case Side.NORTH:
      return (x + w / 2, y + h, Direction.UP)


def _box_endpoint(
  anchor: tuple[int, int],
  side: Side,
  size: tuple[int, int] = (1, 1),
) -> PointAndDirection:
  """Calculates the ending point/direction of a path ending at the given side of the given grid
  bounding box.

  Args:
    anchor: Grid box's anchor point (lower-left).
    side: Side at which the path ends.
    size: Grid box's size (width, height).
  """
  assert len(anchor) == 2
  x, y = anchor
  assert len(size) == 2
  w, h = size
  match side:
    case Side.EAST:
      return (x + w, y + h / 2, Direction.LEFT)
    case Side.SOUTH:
      return (x + w / 2, y, Direction.UP)
    case Side.WEST:
      return (x, y + h / 2, Direction.RIGHT)
    case Side.NORTH:
      return (x + w / 2, y + h, Direction.DOWN)


def _path_piece_points(
  start: PointAndDirection,
  end: PointAndDirection,
  dx: float = 1.,
  dy: float = 1.,
  nudge: float = 0.,
) -> np.ndarray:
  """Constructs the orthogonal path between `start` and `end`.

  Args:
    start: Start point and direction.
    end: End point and direction.
    dx: Horizontal padding (in grid squares) between path and start/end points.
    dy: Vertical padding (in grid squares) between path and start/end points.
    nudge: Offset amount (in grid squares) to nudge path pieces orthogonal to their line directions.
      This breaks the symmetries that often leads to ovelapping paths between grid-aligned points.

  Returns:
    points: Array of path points, shape `(N, 2)`, N >= 2.
  """
  # TODO: Incorporate sizes of start/end elements so that dx/dy can be redefined as margins.

  # To exploit symmetries between different `start`/`end` combinations, convert the problem into
  # a normalized frame by translating/rotating it so `start` becomes `(0, 0, Direction.RIGHT)`,
  # i.e. it is at the origin and the starting path piece points in `Direction.RIGHT`. After
  # constructing the orthogonal path pieces to reach the normalized `end` given by `(x, y, dir)`,
  # transform the path points back to the global frame.
  xs0, ys0, dirs0 = start
  xe0, ye0, dire0 = end
  origin = np.array([xs0, ys0])
  match dirs0:
    case Direction.RIGHT:
      transform = np.array(((1, 0), (0, 1)))  # 0 deg
    case Direction.DOWN:
      transform = np.array(((0, 1), (-1, 0)))  # 270 deg
    case Direction.LEFT:
      transform = np.array(((-1, 0), (0, -1)))  # 180 deg
    case Direction.UP:
      transform = np.array(((0, -1), (1, 0)))  # 90 deg
  dir = _dir_rotate(dire0, dirs0)
  x, y = (np.array([xe0, ye0]) - origin).dot(transform)

  # Choose the path type based on the end position (`x`, `y`) and direction (`dir`). The notation
  # uses path type names (e.g. I1 = 1-segment path shaped like the letter I, C3 = 3-segment path
  # shaped like the letter C, etc.) arranged in a 3x3 grid, where the start position (i.e. origin)
  # is at the center, and the surrounding names indicate the path type based on end position.
  path = []
  if dir == Direction.RIGHT:
    # S5 S5 Z3
    # C5  . I1
    # S5 S5 Z3
    if x >= 0 and y == 0:  # I1
      path = [(0, 0), (x, y)]
    elif x >= 2 * dx:  # Z3
      xmid = x / 2
      path = [(0, 0), (xmid, 0), (xmid, y), (x, y)]
    elif y == 0:  # C5
      path = [(0, 0), (dx, 0), (dx, -dy), (x - dx, -dy), (x - dx, y), (x, y)]
    else:  # S5
      ymid = y / 2
      path = [(0, 0), (dx, 0), (dx, ymid), (x - dx, ymid), (x - dx, y), (x, y)]
  elif dir == Direction.LEFT:
    # C3 C3 C3
    # Q5  . Q5
    # C3 C3 C3
    if abs(y) < dy:  # Q5
      if abs(x) > dx:
        path = [(0, 0), (dx, 0), (dx, -dy), (x + dx, -dy), (x + dx, y), (x, y)]
      else:  # I1 (degenerate)
        path = [(0, 0), (x, y)]
    else:  # C3
      xmax = max(0 + dx, x + dx)
      path = [(0, 0), (xmax, 0), (xmax, y), (x, y)]
  elif dir == Direction.UP:
    # Q4 Q4 L2
    # C4  . Q4
    # C4 C4 Q4
    if x >= dx and y >= dy:  # L2
      path = [(0, 0), (x, 0), (x, y)]
    else:  # Q4/C4
      path = [(0, 0), (dx, 0), (dx, y - dy), (x, y - dy), (x, y)]
  elif dir == Direction.DOWN:
    # C4 C4 Q4
    # C4  . Q4
    # Q4 Q4 L2
    if x >= dx and y <= -dy:  # L2
      path = [(0, 0), (x, 0), (x, y)]
    else:  # Q4/C4
      path = [(0, 0), (dx, 0), (dx, y + dy), (x, y + dy), (x, y)]
  path = np.array(path)
  assert path.shape[1] == 2

  # Nudge all segments while maintaining the same first/last endpoints. There are 3 cases with
  # different nudge compuations depending on the directions of the first and last segments:
  # 1) first = last segment --> horizontal (0, 0)
  # 2) parallel --> horizontal (nudge, 0) ... any (nudge, nudge) ... horizontal (nudge, 0)
  # 3) orthogonal --> horizontal (nudge, 0) ... any (nudge, nudge) ... vertical (0, nudge)
  if path.shape[0] == 2:
    # Case 1: first = last segment.
    pass
  elif dir == Direction.RIGHT or dir == Direction.LEFT:
    # Case 2: parallel segments.
    path[1:-1, 0] += nudge
    path[2:-2, 1] += nudge
  elif dir == Direction.UP or dir == Direction.DOWN:
    # Case 3: orthogonal segments.
    path[1:-2, 0] += nudge
    path[2:-1, 1] += nudge

  # Transform path points back to the global frame.
  return path.dot(transform.T) + origin


def _path_points(
  keypoints: list[PointAndDirection],
  dx: float = 1.,
  dy: float = 1.,
  nudge: float = 0.,
) -> np.ndarray:
  """Constructs the orthogonal path between a sequence of keypoints.

  Args:
    keypoints: List of keypoints, each a `(x, y, direction)` tuple. Must provide >= 2 keypoints.
    dx,dy,nudge: See `_path_piece_points()`.

  Returns:
    points: Array of path points, shape `(M, 2)`, M >= 2.
  """
  assert len(keypoints) >= 2
  pieces = []
  for i in range(len(keypoints) - 1):
    piece = _path_piece_points(keypoints[i], keypoints[i + 1], dx=dx, dy=dy, nudge=nudge)
    # De-duplicate intermediate keypoints.
    if i > 0: piece = piece[1:, :]
    pieces.append(piece)
  return np.vstack(pieces)


def _path_rounded(points: np.ndarray, radius: float = 0.):
  """Rounds the corners of an orthogonal path defined by a sequence of points.

  Args:
    points: Array of path points, shape `(N, 2)`, N >= 1.
    radius: Corner radius (in grid squares). If 0, the path remains orthogonal. If too large for a
      corner, the radius for that corner is automatically reduced.

  Returns:
    vertices,codes: Tuple of path vertices and codes accepted by `matplotlib.path.Path`, with
      vertices shape `(M, 2)`, codes shape `(M,)`, M >= N.
  """
  N = points.shape[0]
  assert N >= 1
  if radius == 0 or N <= 2:
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO for _ in range(N - 1)]
    return np.array(points), np.array(codes)

  verts, codes = [points[0, :]], [mpath.Path.MOVETO]
  for i in range(1, N - 1):
    a, c, b = points[i - 1, :], points[i, :], points[i + 1, :]
    ca = a - c
    cb = b - c
    # Ensure continuous lines defined by start point `a`, corner point `c`, and end point `b`:
    # (1) both have non-zero length, and (2) are orthogonal to each other.
    ca_norm = float(np.linalg.norm(ca))
    cb_norm = float(np.linalg.norm(cb))
    is_valid_corner = ca_norm > 0 and cb_norm > 0 and ca.dot(cb) == 0
    if not is_valid_corner:
      verts.append(c)
      codes.append(mpath.Path.LINETO)
      continue
    # Allow radius to extend at most to half of either line.
    r = min(radius, ca_norm / 2, cb_norm / 2)
    verts.extend([c + r * ca / ca_norm, c, c + r * cb / cb_norm])
    codes.extend([mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3])
  verts.append(points[-1, :])
  codes.append(mpath.Path.LINETO)
  return np.array(verts), np.array(codes)


class OrthogonalPath(renderer.Drawable):
  """A path that connects two grid elements using orthogonal segments."""
  def __init__(
    self,
    ax: maxes.Axes,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    start_side: Side = Side.EAST,
    end_side: Side = Side.WEST,
    start_size: tuple[int, int] = (1, 1),
    end_size: tuple[int, int] = (1, 1),
    waypoints: T.Iterable[PointAndDirection] = [],
    arrow_style: str = '-',
    arrow_kwargs: dict[str, T.Any] = {},
    path_radius: float = 0.5,
    path_nudge: float = 0.,
    linewidth: float = 1.,
    linestyle: str = '-',
    linecolor: str = COLORS.gray,
    pointsize: float = 3.,
    linegrow: float = 1.5,
    value: float = 0.,
  ):
    """
    Args:
      ax: Axes to draw on.
      start: Start grid element's anchor point (lower-left).
      end: End grid element's anchor point (lower-left).
      start_side: Start grid element's side from which the path starts.
      end_side: End grid element's side at which the path ends.
      start_size: Start grid element's size (width, height).
      end_size: End grid element's size (width, height).
      waypoints: Intermediate points/directions through which the path passes.
      arrow_style: Path arrow style. See `matplotlib.patches.ArrowStyle`.
      arrow_kwargs: Path arrow style kwargs. See `matplotlib.patches.ArrowStyle`.
      path_radius: Corner radius for the orthogonal path segments.
      path_nudge: Amount (in grid squares) to offset path pieces othogonally.
      linewidth,linestyle,linecolor: Path styles. See `matplotLib.patches.PathPatch`.
      pointsize: Marker size for waypoints.
      linegrow: Scaling factor for linewidth based on signal value.
      value: Initial signal value.
    """
    super().__init__()
    waypoints = list(waypoints)
    keypoints = [
      _box_startpoint(start, start_side, start_size),
      *waypoints,
      _box_endpoint(end, end_side, end_size),
    ]
    points = _path_points(keypoints, dx=1.0, dy=1.0, nudge=path_nudge)
    verts, codes = _path_rounded(points, path_radius)
    self._path = mpath.Path(verts, codes)
    self._patch = ax.add_patch(
      mpatches.PathPatch(self._path, lw=linewidth, ls=linestyle, ec=linecolor, fill=False)
    )
    # Each element of `paths` is an `mpath.Path` and the corresponding entry in `filled` is a
    # bool indicating whether the `mpatches.PathPatch` should be filled. The first entry is the
    # original path passed in.
    paths, filled = mpatches.ArrowStyle(arrow_style, **arrow_kwargs)(self._path, 1, 0) # type: ignore
    self._arrows = [
      ax.add_patch(
        mpatches.PathPatch(paths[i], lw=linewidth, ec=linecolor, fc=linecolor, fill=filled[i])
      ) for i in range(1, len(paths))
    ]
    self._waypoints = ax.plot(
      [w[0] for w in waypoints],
      [w[1] for w in waypoints],
      '.',
      c=linecolor,
      ms=pointsize,
    )
    # Add signal value.
    self._value = value
    self._linegrow = linegrow
    self._linewidth = linewidth

  def update(self, value: float):
    """Update the signal value, which will modify artists on the next draw.

    Args:
      value: Updated signal value.
    """
    self._value = value

  def draw(self):
    linewidth = (1 + self._linegrow * min(abs(self._value), 1)) * self._linewidth
    self._patch.set_linewidth(linewidth)
    return [self._patch]


__all__ = ['OrthogonalPath', 'Side', 'Direction']
