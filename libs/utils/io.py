import functools
import json
import os
import pathlib
import textwrap
import typing as T

import dill
import numpy as np
import pandas as pd
import termcolor
import yaml

CheckpointData = T.TypeVar('CheckpointData')


def checkpoint(
  filepath: os.PathLike | str,
  func: T.Callable[..., CheckpointData],
  *args,
  cache: bool = False,
  **kwargs,
) -> CheckpointData:
  """Save the result of the function to disk. Optionally, re-executing the function can be skipped
  if its result is already cached."""
  if cache and os.path.exists(filepath):
    with open(filepath, 'rb') as file:
      data = dill.load(file)
      data = T.cast(CheckpointData, data)
    return data
  data = func(*args, **kwargs)
  with open(filepath, 'wb') as file:
    dill.dump(data, file)
  return data


FileData = T.Any
"""Contents of a file, which can be various data formats.."""

DirectoryData = T.Mapping[str, 'FileData | DirectoryData']
"""Contents of a directory, which consists of files and subdirectories."""


class FileRef:
  """Reference to a file. The file data is lazily loaded when `data` is accessed."""

  # TODO: Enable custom loader functions.
  # TODO: Add load(loader, **kwargs), text/binary attribures.
  def __init__(
    self,
    path: pathlib.Path,
    *,
    parent: 'DirectoryRef | None' = None,
    loader: str | None = None,
    **kwargs,
  ):
    self.path = path
    self.parent = parent
    self.loader = loader
    self.kwargs = kwargs

  @property
  def name(self):
    return self.path.name

  @functools.cached_property
  def data(self) -> FileData:
    match self.loader or self.path.suffix:
      case '.txt' | '.log':
        with open(self.path, 'rt') as file:
          return file.read()
      case '.yaml' | '.yml':
        with open(self.path, 'rt') as file:
          return yaml.load(file, yaml.CSafeLoader)
      case '.json':
        with open(self.path, 'rt') as file:
          return json.load(file, **self.kwargs)
      case '.csv':
        return pd.read_csv(self.path, **self.kwargs)
      case '.bin':
        with open(self.path, 'rb') as file:
          return file.read()
      case '.npy' | '.npz':
        return np.load(self.path, **self.kwargs)
      case '.pkl' | '' | _:
        with open(self.path, 'rb') as file:
          return dill.load(file, **self.kwargs)

  def __str__(self):
    return self.path.name

  def __repr__(self):
    return f"FileRef('{self.path.name}')"


class DirectoryRef:
  """Reference to a directory. The directory's (and its subdirectories') file contents are lazily
  loaded when `data` is accessed, up to the `recurse` limit."""

  # TODO: Match include/exclude patterns.
  # TODO: Pattern-based file loaders.
  # TODO: Pretty-printing in Jupyter.
  # TODO: Lazy loading children beyond recurse limit.
  # TODO: Multi-indexing using glob/list in [] operator.
  def __init__(
    self,
    path: pathlib.Path,
    *,
    parent: 'DirectoryRef | None' = None,
    recurse: bool | int = False,
  ):
    self.path = path
    self.parent = parent
    self.children: T.Mapping[str, DirectoryRef | FileRef] = {}

    for child in sorted(self.path.glob('*')):
      if child.name.startswith('.'):
        continue
      elif child.is_file():
        self.children[child.name] = FileRef(child)
      elif recurse and child.is_dir():
        # For historical reasons, `bool` is a subclass of `int`. So the typecheck must be done on
        # the more specific class, i.e. `bool`.
        self.children[child.name] = DirectoryRef(
          child,
          parent=self,
          recurse=recurse if isinstance(recurse, bool) else recurse - 1,
        )

  @property
  def name(self):
    return self.path.name

  @property
  def data(self) -> DirectoryData:
    return {k: v.data for k, v in self.children.items()}

  def __getitem__(self, key: str | pathlib.Path):
    if isinstance(key, str):
      key = pathlib.Path(key)
    first, *rest = key.parts
    if first in self.children:
      child = self.children[first]
      if isinstance(child, FileRef):
        if len(rest) > 0:
          raise KeyError(f'Cannot index file as a directory: {key}')
        return child
      elif len(rest) == 0:
        return child
      else:
        return child[pathlib.Path(*rest)]
    raise KeyError(f'Directory "{self.path}" does not have child: {first}')

  def __len__(self):
    return len(self.children)

  def __iter__(self):
    return iter(self.children)

  def __str__(self):
    title = termcolor.colored(self.path.absolute().name, attrs=['bold'])
    children = textwrap.indent('\n'.join([str(child) for child in self.children.values()]), '│ ')
    return f'{title}\n{children}' if children else title

  def __repr__(self):
    title = termcolor.colored(f"DirectoryRef('{self.path.absolute().name}')", attrs=['bold'])
    children = textwrap.indent('\n'.join([repr(child) for child in self.children.values()]), '│ ')
    return f'{title}\n{children}' if children else title


@T.overload
def read_file(
  filepath: os.PathLike | str,
  *,
  lazy: T.Literal[False] = ...,
  loader: str | None = ...,
  **kwargs,
) -> FileData:
  ...


@T.overload
def read_file(
  filepath: os.PathLike | str,
  *,
  lazy: T.Literal[True] = ...,
  loader: str | None = ...,
  **kwargs,
) -> FileRef:
  ...


def read_file(filepath, *, lazy=False, loader=None, **kwargs):
  """Read a single file from disk. If `lazy=True`, a `FileRef` object is returned instead of the
  file data itself."""
  filepath = pathlib.Path(os.path.expandvars(os.path.expanduser(filepath)))
  ref = FileRef(filepath, loader=loader, **kwargs)
  return ref if lazy else ref.data


@T.overload
def read_files(
  dirpath: os.PathLike | str,
  *,
  lazy: T.Literal[False] = ...,
  recurse: bool | int = ...,
) -> DirectoryData:
  ...


@T.overload
def read_files(
  dirpath: os.PathLike | str,
  *,
  lazy: T.Literal[True] = ...,
  recurse: bool | int = ...,
) -> DirectoryRef:
  ...


def read_files(dirpath, *, lazy=False, recurse: bool | int = False):
  """Read files in a directory from disk. If `lazy=True`, a `DirectoryRef` object is returned
  instead of the directory data itself."""
  # TODO: Add include/exclude glob patterns.
  dirpath = pathlib.Path(os.path.expandvars(os.path.expanduser(dirpath)))
  if isinstance(recurse, int) and recurse < 0:
    raise ValueError(f'Invalid recurse (must be non-negative): {recurse}')
  ref = DirectoryRef(dirpath, recurse=recurse)
  return ref if lazy else ref.data


def write_file(
  filepath: os.PathLike | str,
  data: FileData,
  reload: bool = False,
) -> FileData:
  """Write a single file to disk."""
  filepath = pathlib.Path(os.path.expandvars(os.path.expanduser(filepath)))
  filepath.parent.mkdir(parents=True, exist_ok=True)

  match filepath.suffix:
    case '.txt':
      with open(filepath, 'wt') as file:
        file.write(data)  # type: ignore
    case '.yaml' | '.yml':
      with open(filepath, 'wt') as file:
        yaml.dump(data, file, yaml.CSafeDumper, sort_keys=False)
    case '.json':
      raise NotImplementedError()
    case '.csv':
      pd.DataFrame(data, copy=False).to_csv(filepath)
    case '.bin':
      with open(filepath, 'wb') as file:
        file.write(data)  # type: ignore
    case '.npy':
      raise NotImplementedError()
    case '.npz':
      raise NotImplementedError()
    case '.pkl' | '' | _:
      with open(filepath, 'wb') as file:
        dill.dump(data, file)

  return read_file(filepath, lazy=False) if reload else data


def write_files(
  dirpath: os.PathLike | str,
  data: dict[str, T.Any],
) -> DirectoryRef:
  """Write files and subdirectories to disk."""
  raise NotImplementedError()


# Access One:
# files['foo/bar']
# files['foo']['bar']
# files['bar'].parent
# files['bar/..'] == files['bar']['..'] == files['bar'].parent == files['foo']

# Access Multiple:
# files[['foo', 'foo/bar']] == [data1, data2]

# Glob:
# files['f*'] == { 'foo': data1 }
# files[['foo/*', 'bar/*']] == { 'foo/1': data1, 'foo/2': data2, 'bar/3': data3 }

# Exists:
# files.exists('foo/dw')

# Append to file (e.g. csv log).
# Use pyfilesystem2.
