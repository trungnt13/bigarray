from __future__ import absolute_import, division, print_function

import os
import unittest
import zlib
from io import BytesIO
from multiprocessing import Pool
from tempfile import mkstemp

import numpy as np

from bigarray import MmapArray, MmapArrayWriter

np.random.seed(8)


# ===========================================================================
# Helper function
# ===========================================================================
def _get_tempfile():
  fid, fpath = mkstemp()
  os.close(fid)
  return fpath


def _fn_write(job):
  idx, array, path, shape = job
  with MmapArrayWriter(path=path, shape=shape, dtype='float64') as f:
    f.write(array, start_position=idx * array.shape[0])


def _fn_read(job):
  marray, (start, end) = job
  data = marray[start:end].tobytes()
  return (start, end, zlib.compress(data))


# ===========================================================================
# Test cases
# ===========================================================================
class MmapArrayTest(unittest.TestCase):

  def test_write_single_time(self):
    fpath = _get_tempfile()
    array = np.arange(0, 100, dtype='float32').reshape(-1, 5)

    with MmapArrayWriter(path=fpath,
                         shape=array.shape,
                         dtype=array.dtype,
                         remove_exist=True) as f:
      f.write(array)
    x = MmapArray(fpath)
    self.assertTrue(np.all(array == x))

    with MmapArrayWriter(path=fpath, remove_exist=False) as f:
      f.write(array)
    x = MmapArray(fpath)
    self.assertTrue(np.all(np.concatenate([array, array], axis=0) == x))

  def test_write_multiple_time(self):
    fpath = _get_tempfile()
    array = np.arange(0, 1000, dtype='float32').reshape(-1, 2, 5)

    with MmapArrayWriter(path=fpath,
                         shape=(0,) + array.shape[1:],
                         dtype=array.dtype,
                         remove_exist=True) as f:
      for i in range(0, array.shape[0], 8):
        f.write(array[i:i + 8])
    x = MmapArray(fpath)
    self.assertTrue(np.all(array == x))

    array1 = np.arange(0, 100, dtype='float32').reshape(-1, 2, 5)
    array[10:10 + array1.shape[0]] = array1
    with MmapArrayWriter(path=fpath, remove_exist=False) as f:
      f.write(array1, start_position=10)
    x = MmapArray(fpath)
    self.assertTrue(np.all(array == x))

  def test_write_multiprocessing(self):
    fpath = _get_tempfile()
    jobs = [
        (i, np.random.rand(12, 25, 8), fpath, (300, 25, 8)) for i in range(25)
    ]
    with Pool(2) as pool:
      pool.map(_fn_write, jobs)

    # checking the output
    array = np.concatenate([x[1] for x in jobs], axis=0)
    x = MmapArray(fpath)
    self.assertTrue(np.all(array == x))

  def test_read_multiprocessing(self):
    fpath = _get_tempfile()
    array = np.random.rand(1200, 25, 8)
    # first write the array
    with MmapArrayWriter(fpath, (None, 25, 8), array.dtype) as f:
      f.write(array)
    x = MmapArray(fpath)
    self.assertTrue(np.all(array == x))
    # use multiprocessing to randomly read the array
    jobs = [
        (x,
         sorted(np.random.randint(0, array.shape[0], size=(2,), dtype='int32')))
        for i in range(25)
    ]
    with Pool(2) as pool:
      for start, end, data in pool.map(_fn_read, jobs):
        data = zlib.decompress(data)
        data = np.frombuffer(data).reshape(-1, 25, 8)
        self.assertTrue(np.all(data == array[start:end]))


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  unittest.main()
