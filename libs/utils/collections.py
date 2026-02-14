import collections
import typing as T

import utils.string

K = T.TypeVar('K')
V = T.TypeVar('V')


class AttrDict(dict[str, T.Any]):
  """A dictionary that allows access to its keys as attributes."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self


class GlobDict(dict[str, T.Any]):
  """A dictionary mapping glob patterns to values. Using specific (non-pattern) keys will match
  glob patterns for key membership and value access.

    ```
    >>> d = GlobDict({'a*': 1})
    >>> 'apple' in d
    True
    >>> d['apple']
    1
    >>> d['banana']
    Traceback (most recent call last):
      ...
    KeyError: 'banana'
    >>> d.get('banana', 2)
    2
    ```
  """
  def __contains__(self, key: str) -> bool:
    return super().__contains__(key) or utils.string.in_glob_range(self.keys(), key)

  def __getitem__(self, key: str) -> object:
    return utils.string.select_by_glob_range(self, key)

  def get(self, key: str, default: T.Any = None) -> T.Any:
    return utils.string.select_by_glob_range(self, key, default)


def zip_dict(iterable_of_dicts: T.Iterable[dict[K, V]]) -> dict[K, list[V]]:
  """Zips an iterable of dictionaries into a dictionary of lists. The keys of the output are the
  union of the keys of the input. The values of the output are lists of the corresponding values in
  the input. The value lists are ordered as in the input, and they may be of different lengths if
  keys are missing in the input."""
  out = collections.defaultdict(list)
  for d in iterable_of_dicts:
    for k, v in d.items():
      out[k].append(v)
  return out


__all__ = ['AttrDict', 'GlobDict', 'zip_dict']
