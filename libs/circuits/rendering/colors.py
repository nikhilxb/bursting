"""
This module defines the color palette used for rendering.
"""
import dataclasses


# Mapping[color name string, color hex string].
# yapf: disable
@dataclasses.dataclass(frozen=True)
class ColorPalette:
  """Mapping of color names to hex strings."""
  blue      = '#0975C1'
  bluel     = '#C7E8F7'
  red       = '#CB4137'
  redl      = '#FEDBD4'
  yellow    = '#E3A855'
  yellowl   = '#FAEFC5'
  green     = '#06A66C'
  greenl    = '#C4ECD5'
  purple    = '#9A53BE'
  purplel   = '#EEDDF4'
  orange    = '#DD743E'
  orangel   = '#FFDCC3'
  cyan      = '#23C1BB'
  cyanl     = '#C3EAE7'
  magenta   = '#E163B1'
  magental  = '#F8DAEA'
  lime      = '#8BBC64'
  limel     = '#E1F1CC'
  indigo    = '#8388E4'
  indigol   = '#DFE1FB'
  graydd    = '#3E3E3E'
  grayd     = '#727272'
  gray      = '#AEAEAE'
  grayl     = '#E2E2E2'
  grayll    = '#EDEDED'
  white     = '#FFFFFF'
  black     = '#000000'
# yapf: disable

COLORS = ColorPalette()
"""Default color palette."""
