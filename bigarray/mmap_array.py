from __future__ import absolute_import, division, print_function

import marshal
import os
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterable, List, Optional, Text, Tuple, Union

import numpy as np
from six import string_types

__all__ = [
    'get_total_opened_mmap',
    'read_mmaparray_header',
    'MmapArrayWriter',
    'MmapArray',
]

MAX_OPEN_MMAP = 120
_INSTANCES_WRITER = OrderedDict()
_HEADER = b'mmapdata'
_HEADER_SIZE_LENGTH = 8
_MAXIMUM_HEADER_SIZE = 486


# ===========================================================================
# Helper
# ===========================================================================
def get_total_opened_mmap():
  return len(_INSTANCES_WRITER)


def read_mmaparray_header(path, return_header_size=False):
  """ Reading header (if available) of a MmapArray

  Parameters
  ----------
  path : `str`
    Input path to a file

  Return
  ------
  dtype, shape
    Necessary information to create numpy.memmap
  """
  header_size = 0
  with open(path, mode='rb') as f:
    # ====== check header signature ====== #
    if f.read(len(_HEADER)) != _HEADER:
      raise ValueError('Invalid header for MmapData.')
    header_size += len(_HEADER)
    # ====== 8 bytes for size of info ====== #
    try:
      size = f.read(_HEADER_SIZE_LENGTH)
      header_size += len(size)

      metadata = f.read(int(size))
      header_size += len(metadata)

      dtype, shape = marshal.loads(metadata)
    except Exception as e:
      f.close()
      raise Exception('Error reading memmap data file: %s' % str(e))
    # ====== return file object ====== #
    if return_header_size:
      return dtype, shape, header_size
    return dtype, shape


def _aligned_memmap_offset(dtype):
  header_size = len(_HEADER) + 8 + _MAXIMUM_HEADER_SIZE
  type_size = np.dtype(dtype).itemsize
  n = np.ceil(header_size / type_size)
  return int(n * type_size)


# ===========================================================================
# Writing new memory-mapped array
# ===========================================================================
class MmapArrayWriter(object):
  """ Helper class for writing data to `MmapArray`, this class is singleton,
  i.e. there are never two instance point to the same path

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

  def __new__(cls, path, *args, **kwargs):
    # an absolute path would give stronger identity
    if isinstance(path, string_types):
      path = os.path.abspath(path)
    # file id is given
    else:
      raise ValueError("Only support file path, and not file descriptor ID")
    # Found old instance
    if path in _INSTANCES_WRITER:
      obj = _INSTANCES_WRITER[path]
      if not obj.is_closed:
        return obj
    # ====== increase memmap count ====== #
    if get_total_opened_mmap() > MAX_OPEN_MMAP:
      raise RuntimeError("Only allowed to open maximum of %d memmap file" %
                         MAX_OPEN_MMAP)
    # ====== create new instance ====== #
    new_instance = super(MmapArrayWriter, cls).__new__(cls)
    _INSTANCES_WRITER[path] = new_instance
    return new_instance

  def __init__(self,
               path: Text,
               shape: Optional[List[int]] = None,
               dtype: Optional[Union[Text, np.dtype]] = None,
               remove_exist: bool = False):
    super(MmapArrayWriter, self).__init__()
    if isinstance(path, string_types):
      # validate path
      path = os.path.abspath(path)
      # remove exist
      if remove_exist and os.path.exists(path):
        if os.path.isfile(path):
          os.remove(path)
        else:
          raise RuntimeError("Give path at '%s' is a folder, cannot remove!" %
                             path)
    else:
      raise ValueError("Only support file path, and not file descriptor ID")
    # ====== read exist file ====== #
    if os.path.exists(path) and os.stat(path).st_size > 0:
      dtype, shape, self._header_size = read_mmaparray_header(
          path, return_header_size=True)
      f = open(path, 'rb+')
      self._start_position = shape[0]
    # ====== create new file ====== #
    else:
      self._start_position = 0
      self._header_size = 0
      if dtype is None or shape is None:
        raise Exception("First created this MmapData, `dtype` and "
                        "`shape` must NOT be None.")
      # check shape info
      if not isinstance(shape, Iterable):
        shape = (shape,)
      shape = tuple([0 if i is None or i < 0 else int(i) for i in shape])
      # open the file
      f = open(path, 'wb+')
      f.write(_HEADER)
      self._header_size += len(_HEADER)
      # save dtype and shape to the header
      dtype = str(np.dtype(dtype))
      if isinstance(shape, np.ndarray):
        shape = shape.tolist()
      if not isinstance(shape, (tuple, list)):
        shape = (shape,)
      # write header metadata
      # TODO: not a good solution, this could course overflow when expanding
      # the memmap data, but if modifying the header algorithm,
      # it is backward incompatible
      header_meta = marshal.dumps([dtype, shape])
      size = len(header_meta)
      if size > _MAXIMUM_HEADER_SIZE:
        raise Exception('The size of header excess maximum allowed size '
                        '(%d bytes).' % _MAXIMUM_HEADER_SIZE)
      header_length = ('%8d' % size).encode()
      f.write(header_length)
      self._header_size += len(header_length)
      f.write(header_meta)
      self._header_size += len(header_meta)
    # ====== assign attributes ====== #
    self._file = f
    self._path = path if isinstance(path, string_types) else \
      f.name
    data = np.memmap(f,
                     dtype=dtype,
                     shape=shape,
                     mode='r+',
                     offset=_aligned_memmap_offset(dtype))
    self._data = data
    self._is_closed = False

  def __getstate__(self):
    raise NotImplementedError

  def __setstate__(self, states):
    raise NotImplementedError

  @property
  def filesize(self):
    """ Return the size of the mmap file in bytes """
    return os.stat(self.path).st_size

  @property
  def shape(self):
    return self._data.shape

  @property
  def path(self):
    return self._path

  @property
  def is_closed(self):
    return self._is_closed

  def flush(self):
    self._data.flush()

  def close(self):
    if self.is_closed:
      return
    self._is_closed = True
    if self.path in _INSTANCES_WRITER:
      del _INSTANCES_WRITER[self.path]
    # flush in read-write mode
    self.flush()
    # close mmap and file
    self._data._mmap.close()
    del self._data
    self._file.close()
    del self._file

  def _resize(self, new_length):
    # ====== local files ====== #
    f = self._file
    old_length = self._data.shape[0]
    # ====== check new shape ====== #
    if new_length < old_length:
      raise ValueError(
          'Only support extending memmap, and cannot shrink the memory.')
    # nothing to resize
    elif new_length == old_length:
      return self
    # ====== flush previous changes ====== #
    # resize by create new memmap and also rename old file
    shape = (new_length,) + self._data.shape[1:]
    dtype = self._data.dtype.name
    # rewrite the header, update metadata
    f.seek(len(_HEADER))
    meta = marshal.dumps([dtype, shape])
    size = '%8d' % len(meta)
    f.write(size.encode(encoding='utf-8'))
    f.write(meta)
    f.flush()
    # close old data
    self._data._mmap.close()
    del self._data
    # extend the memmap, by open numpy.memmap with bigger shape
    f = open(self.path, 'rb+')
    self._file = f
    mmap = np.memmap(f,
                     dtype=dtype,
                     shape=shape,
                     mode='r+',
                     offset=_aligned_memmap_offset(dtype))
    self._data = mmap
    return self

  def write(self, arrays: Iterable, start_position=None):
    """ Extending the memory-mapped data and copy the array
    into extended area.

    Parameters
    ----------
    arrays : {`numpy.ndarray`, list of `numpy.ndarray`}
      multiple arrays could be written to the file at once for optimizing
      the performance.
    start_position {`None`, `int`}
      if `None`, appending the data to the `MmapArray`
      if a positive integer is given, write the data start from given position
      if a negative integer is given, write the data from `end - start_position`

    Return
    ------
    `MmapArrayWriter` for method chaining
    """
    if self.is_closed:
      raise RuntimeError("The MmapArrayWriter is closed!")
    # only get arrays matched the shape
    add_size = 0
    if not isinstance(arrays, Iterable) or isinstance(arrays, np.ndarray):
      arrays = (arrays,)
    # ====== check if shape[1:] matching ====== #
    accepted_arrays = []
    for a in arrays:
      if a.shape[1:] == self._data.shape[1:]:
        accepted_arrays.append(a)
        add_size += a.shape[0]
    # no new array to append
    if len(accepted_arrays) == 0:
      raise RuntimeError("No appropriate array found for writing, given: %s, "
                         "; but require array with shape: %s" %
                         (','.join([str(i.shape) for i in arrays]), self.shape))
    # ====== resize ====== #
    if start_position is None:
      start_position = self._start_position
      given_start_position = False
    else:
      start_position = int(start_position)
      given_start_position = True
      if start_position < 0:
        start_position = self.shape[0] - start_position
    add_length = add_size - (self.shape[0] - start_position)
    # resize and append data (resize only once will be faster)
    if add_length > 0:
      self._resize(self.shape[0] + add_length)
    # ====== update values ====== #
    data = self._data
    for a in accepted_arrays:
      data[start_position:start_position + a.shape[0]] = a
      start_position += a.shape[0]
    if not given_start_position:
      self._start_position = start_position
    return self

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()

  def __del__(self):
    self.close()


# ===========================================================================
# Reading the array
# ===========================================================================
class MmapArray(np.memmap):
  """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

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

  def __new__(subtype, path, mode='r+'):
    if isinstance(path, string_types):
      path = os.path.abspath(path)
      if not os.path.exists(path) and os.path.isfile(path):
        raise ValueError(
            'path must be existed file created by MmapArrayWriter.')
    else:
      raise ValueError("Only support file path, and not file descriptor ID")
    dtype, shape = read_mmaparray_header(path)
    offset = _aligned_memmap_offset(dtype)
    new_array = super(MmapArray, subtype).__new__(subtype=subtype,
                                                  filename=path,
                                                  dtype=dtype,
                                                  mode=mode,
                                                  offset=offset,
                                                  shape=shape)
    new_array._path = path
    return new_array

  @property
  def path(self):
    return self._path

  @property
  def filesize(self):
    """ Return the size of the mmap file in bytes """
    return os.stat(self.path).st_size
