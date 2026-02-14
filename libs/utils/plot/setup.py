import os
import typing as T

import matplotlib.colors as mcolors
import matplotlib.font_manager as mfonts
import proplot as pplt


def setup_fonts(fontpaths: T.Iterable[str | os.PathLike], verbose: bool = False, **kwargs):
  """Register fonts from the given paths."""
  fontpaths = [os.path.expandvars(os.path.expanduser(path)) for path in fontpaths]
  for font in sorted(mfonts.findSystemFonts(fontpaths=fontpaths, **kwargs)):
    mfonts.fontManager.addfont(font)
    if verbose: print(f'font:\t{os.path.basename(font)}')


def setup_colors(colorpaths: T.Iterable[str | os.PathLike], verbose: bool = False):
  """Register colors from the given theme files. Each line in a theme file should define the color
  hex code followed by the color name/aliases. At least one name is required. Blank lines and
  comment lines (starting with `#`) are ignored.
    ```
    COLOR_HEX NAME ALIAS1 ALIAS2 ...
    000000 black k
    ```
  """
  colorpaths = [os.path.expandvars(os.path.expanduser(path)) for path in colorpaths]
  for colorpath in colorpaths:
    if not os.path.isfile(colorpath): continue
    with open(colorpath, 'rt') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        hex, *names = line.split()
        if len(names) == 0: continue
        for name in names:
          mcolors._colors_full_map[name] = f'#{hex}'  # type: ignore
        if verbose: print(f'color:\t{hex} {" ".join(names)}')


def setup_plotting(
  *,
  fonts: T.Iterable[str | os.PathLike] = ['$REPO_ROOT/assets/fonts'],
  colors: T.Iterable[str | os.PathLike] = ['$REPO_ROOT/assets/styles/scientific.txt'],
  cmaps: T.Iterable[str | os.PathLike] = ['$REPO_ROOT/assets/styles/cmaps'],
  mplstyle: T.Iterable[str | os.PathLike] = ['$REPO_ROOT/assets/styles/scientific.mplstyle'],
  mplstyle_kws: dict[str, T.Any] = { 'figure.dpi': 144, 'figure.facecolor': 'white'},
  proplotrc: str | os.PathLike = '$REPO_ROOT/assets/styles/scientific.proplotrc',
  verbose: bool = False,
):
  setup_fonts(fonts, verbose=verbose)
  setup_colors(colors, verbose=verbose)

  # cmaps = [os.path.expandvars(os.path.expanduser(x)) for x in cmaps]
  # mplstyle = [os.path.expandvars(os.path.expanduser(x)) for x in mplstyle]
  # proplotrc = os.path.expandvars(os.path.expanduser(proplotrc))
  # pplt.register_cmaps(*cmaps)  # type: ignore
  # pplt.use_style(mplstyle + [mplstyle_kws])  # type: ignore
  # pplt.rc.load(proplotrc)  # type: ignore
