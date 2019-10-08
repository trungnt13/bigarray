from __future__ import absolute_import, division, print_function

import os
import pickle
from collections import OrderedDict
from multiprocessing import Lock, Manager, Value
from typing import Dict, Iterable, List, Optional, Text, Tuple, Union

import numpy as np
from six import string_types

from bigarray.mmap_array import (_HEADER, _MAXIMUM_HEADER_SIZE, MmapArray,
                                 MmapArrayWriter)

__all__ = ['PointerArrayWriter', 'PointerArray']


# ===========================================================================
# Helper
# ===========================================================================
class _ReadOnlyDict(dict):

  def __readonly__(self, *args, **kwargs):
    raise RuntimeError(
        "Cannot modify the indices of PointerArray, "
        "use PointerArrayWriter if you want to modify the contents.")

  __setitem__ = __readonly__
  __delitem__ = __readonly__
  pop = __readonly__
  popitem = __readonly__
  clear = __readonly__
  update = __readonly__
  setdefault = __readonly__
  del __readonly__


class _SharedDictWriter(object):
  """ A multiprocessing syncrhonized dictionary for writing """

  def __init__(self, d):
    assert isinstance(d, dict)
    self._manager = Manager()
    self._dict = self._manager.dict(d)
    self._lock = Lock()
    self._main_pid = os.getpid()
    self._is_multiprocessing = Value('i', 0)

  def update(self, items):
    with self._lock:
      if os.getpid() != self._main_pid:
        self._is_multiprocessing.value += 1
      self._dict.update(items)

  @property
  def values(self):
    with self._lock:
      return self._dict

  def dispose(self):
    with self._lock:
      if self._is_multiprocessing.value > 0:
        pass
    del self._manager
    del self._lock
    del self._dict


# ===========================================================================
# PointerArray
# ===========================================================================
class PointerArrayWriter(MmapArrayWriter):
  """ Helper class for writing data to `PointerArray`, this class is singleton,
  i.e. there are never two instance point to the same path

  The `PointerArrayWriter` support multiprocessing writing on the indices
  that mapping from an identity (string type) to a tuple of `(start, end)`
  position within the array

  Parameters
  ----------
  path : str
    path to a file for writing memory-mapped data
  shape : `tuple`
    tuple of integer representing the shape
  dtype : numpy.dtype
    data type
  remove_exist : boolean (default=False)
    if file at given path exists, remove it
  """

  def __init__(self,
               path: Text,
               shape: Optional[List[int]] = None,
               dtype: Optional[Union[Text, np.dtype]] = None,
               remove_exist: bool = False):
    super(PointerArrayWriter, self).__init__(path=path,
                                             shape=shape,
                                             dtype=dtype,
                                             remove_exist=remove_exist)
    self._is_indices_saved = False
    # first time create the file
    if self._start_position == 0 or self.shape[0] == 0:
      self._indices = _SharedDictWriter(OrderedDict())
    # MmapArray already existed
    else:
      cur_pos = self._file.tell()
      filesize = os.stat(self.path).st_size
      self._file.seek(filesize - 8)
      indices_size = int.from_bytes(self._file.read(8), 'big')
      self._file.seek(filesize - 8 - indices_size)
      self._indices = _SharedDictWriter(
          pickle.loads(self._file.read(indices_size)))
      self._file.seek(cur_pos)

  @property
  def indices(self):
    return _ReadOnlyDict(self._indices.values)

  def write(self, arrays: Dict[Text, np.ndarray], start_position=None):
    """ Extending the memory-mapped data and copy the array
    into extended area.

    Parameters
    ----------
    arrays : `dict`
      a mapping from `str` to `numpy.ndarray`
    start_position {`None`, `int`}
      if `None`, appending the data to the `MmapArray`
      if a positive integer is given, write the data start from given position
      if a negative integer is given, write the data from `end - start_position`

    Return
    ------
    `PointerArrayWriter` for method chaining
    """
    self._is_indices_saved = False
    if not isinstance(arrays, dict) and all(
        isinstance(i, string_types) and hasattr(j, 'shape')
        for i, j in arrays.items()):
      raise ValueError("write function only accept dictionary mapping from "
                       "a text identity to numpy.ndarray")
    items = list(arrays.items())
    names = [i[0] for i in items]
    arrays = [i[1] for i in items]
    indices = {}
    # ====== creating the indices ====== #
    if start_position is None:
      start = self._start_position
    else:
      start = int(start_position)
      if start < 0:
        start = self.shape[0] - start

    accepted_names = []
    accepted_arrays = []
    for n, a in zip(names, arrays):
      if a.shape[1:] == self._data.shape[1:]:
        accepted_names.append(n)
        accepted_arrays.append(a)

    for name, a in zip(accepted_names, accepted_arrays):
      indices[name] = (start, start + a.shape[0])
      start += a.shape[0]
    self._indices.update(indices)
    # ====== writing the arrays ====== #
    return super(PointerArrayWriter, self).write(accepted_arrays,
                                                 start_position)

  def _flush_indices(self):
    if self.is_closed or self._is_indices_saved:
      return
    self._is_indices_saved = True
    with open(self.path, 'ab') as f:
      indices = dict(self._indices.values)
      indices_data = pickle.dumps(indices)
      size = len(indices_data).to_bytes(8, 'big')
      f.write(indices_data + size)

  def close(self):
    self._flush_indices()
    if not self.is_closed:
      self._indices.dispose()
    super(PointerArrayWriter, self).close()


class PointerArray(MmapArray):
  """Create a memory-map to an array stored in a *binary* file on disk.

    The `PointerArray` is different from `MmapArray` in away that, it contains
    a dictionary mapping from an identity (string type) to start and end
    position within the array.

    Delete the memmap instance to close the memmap file.

    Parameters
    ----------
    path : str, file-like object, or pathlib.Path instance
        The file name or file object to be used as the array data buffer.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+
        Default is 'r+'.
  """

  def __init__(self, *args, **kwargs):
    super(PointerArray, self).__init__()
    with open(self.path, 'rb') as f:
      filesize = os.stat(self.path).st_size
      f.seek(filesize - 8)
      indices_size = int.from_bytes(f.read(8), 'big')
      f.seek(filesize - 8 - indices_size)
      self._indices = _ReadOnlyDict(pickle.loads(f.read(indices_size)))

  @property
  def indices(self):
    return self._indices

  def __getitem__(self, key):
    if isinstance(key, string_types):
      start, end = self._indices[key]
      return self[start:end]
    return super(PointerArray, self).__getitem__(key)
