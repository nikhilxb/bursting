"""
This module defines the `Cell` classes, which are drawables that represent circuit elements.
"""
import typing as T

import matplotlib.axes as maxes
import matplotlib.patches as mpatches

from circuits.rendering import colors, renderer

COLORS = colors.COLORS


class SignalCell(renderer.Drawable):
  """A cell that represents a signal through the height of an activity bar."""
  def __init__(
    self,
    ax: maxes.Axes,
    x: int,
    y: int,
    value: float = 0.,
    label: bool = True,
    bgcolor: str = COLORS.grayll,
    fgcolor: str | T.Literal['sign'] | T.Callable[[float], str] = 'sign',
  ):
    """
    Args:
      ax: Axes to draw on.
      x: X-coordinate of the cell's anchor point (lower-left).
      y: Y-coordinate of the cell's anchor point (lower-left).
      value: Initial signal value.
      label: Whether to label the signal value.
      bgcolor: Background color.
      fgcolor: Foreground color. If 'sign', the color is red for positive values and blue for
        negative values. If a callable, the color is dynamically determined by the callable based
        on the signal value. Otherwise, the color is constant.
    """
    self.x = x
    self.y = y
    self.value = value
    self.label = label

    if callable(fgcolor):
      self.fgcolor = fgcolor
    elif fgcolor == 'sign':
      self.fgcolor = lambda v: COLORS.red if v > 0 else COLORS.blue if v < 0 else bgcolor
    else:
      self.fgcolor = lambda v: fgcolor

    self._bg = ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=bgcolor))
    self._fg = ax.add_patch(mpatches.Rectangle((x, y), 1, 0, facecolor=bgcolor))
    if self.label:
      self._text = ax.text(x + 0.5, y, '', ha='center', c=COLORS.graydd, size='x-small')

  def update(self, value: float):
    """Update the signal value, which will modify artists on the next draw.

    Args:
      value: Updated signal value.
    """
    self.value = value

  def draw(self):
    value = self.value
    height = min(abs(value), 1.)
    self._fg.set(
      height=height,
      facecolor=self.fgcolor(value),
    )
    if self.label:
      self._text.set(text=f'{value:3.1f}')
      return [self._fg, self._text]
    else:
      return [self._fg]


__all__ = ['SignalCell']
