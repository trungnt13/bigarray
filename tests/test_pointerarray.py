from __future__ import absolute_import, division, print_function

import os
import pickle
import unittest
import zlib
from multiprocessing import Pool
from tempfile import mkstemp

import numpy as np

from bigarray import PointerArray, PointerArrayWriter

np.random.seed(8)

# ===========================================================================
# Helper
# ===========================================================================
WRITER = None


def _get_tempfile():
  fid, fpath = mkstemp()
  os.close(fid)
  return fpath


def _del_file(path):
  try:
    if os.path.exists(path):
      os.remove(path)
  except Exception as e:
    pass


def _fn_write(job):
  start_position, arrays = job
  WRITER.write(arrays, start_position=start_position)


def _fn_write2(job):
  writer, arrays = job
  start_position = min(np.min(i) for i in arrays.values())
  writer.write(arrays, start_position=start_position)


def _fn_read(job):
  names, path = job
  x = PointerArray(path)
  data = {i: (x[i].tobytes(), x[i].dtype, x[i].shape) for i in names}
  return zlib.compress(pickle.dumps(data))


# ===========================================================================
# Test cases
# ===========================================================================
class PointerArrayTest(unittest.TestCase):

  def test_writing_pointerarray(self):
    path = _get_tempfile()
    data = {'name%d' % i: np.random.rand(12, 8) for i in range(2500)}
    data1 = {'name%d' % i: np.random.rand(12, 8) for i in range(2500, 5000)}

    with PointerArrayWriter(path,
                            shape=(0, 8),
                            dtype='float64',
                            remove_exist=True) as f:
      f.write(data)

    with PointerArrayWriter(path, remove_exist=False) as f:
      f.write(data1)

    x = PointerArray(path)

    is_data_matching = True
    for name, (start, end) in x.indices.items():
      dat = data[name] if name in data else data1[name]
      y = x[start:end]
      is_data_matching &= np.all(dat == y)
    self.assertTrue(is_data_matching)

    _del_file(path)

  def test_reading_writing_multiprocessing(self):
    path = _get_tempfile()

    # generate jobs (make sure no overlap)
    jobs = []
    n = 0
    for job_idx in range(25):
      job = {}
      start = n
      for i in range(250):
        x = np.random.rand(np.random.randint(8, 12, dtype='int32'), 8)
        n += x.shape[0]
        job['name_%d_%d' % (job_idx, i)] = x
      jobs.append((start, job))

    # multiprocess writing
    global WRITER
    WRITER = PointerArrayWriter(path,
                                shape=(n, 8),
                                dtype='float64',
                                remove_exist=True)

    with Pool(2) as pool:
      pool.map(_fn_write, jobs)
    WRITER.flush()
    WRITER.close()

    # checking the data
    x = PointerArray(path)
    all_names = []
    all_data = {}
    for _, arrays in jobs:
      for name, dat in arrays.items():
        # store for later read test
        all_names.append(name)
        all_data[name] = dat
        # check the data
        y = x[name]
        self.assertTrue(np.all(dat == y))
    del x

    # multiprocessing list
    jobs = [(j, path) for j in np.array_split(all_names, 8)]
    with Pool(2) as pool:
      for data in pool.map(_fn_read, jobs):
        data = pickle.loads(zlib.decompress(data))
        data = {
            name: np.frombuffer(dat, dtype).reshape(shape)
            for name, (dat, dtype, shape) in data.items()
        }
        self.assertTrue(
            all(np.all(dat == all_data[name]) for name, dat in data.items()))
    _del_file(path)

  def test_pickling(self):
    path = _get_tempfile()

    f = PointerArrayWriter(path, shape=(0,), dtype='float64', remove_exist=True)
    f.write({
        'name0': np.arange(0, 10),
        'name1': np.arange(10, 20),
    })

    f1 = pickle.loads(pickle.dumps(f))
    f1.write({
        'name2': np.arange(20, 30),
        'name3': np.arange(30, 40),
    })
    # the order is important, if `f` flush later, the order will change
    f.flush()
    f1.flush()

    x = PointerArray(path)
    self.assertTrue(np.all(x['name0'] == np.arange(0, 10)))
    self.assertTrue(np.all(x['name3'] == np.arange(30, 40)))

  def test_pickling_multiprocessing(self):
    path = _get_tempfile()
    f = PointerArrayWriter(path,
                           shape=(25 * 10,),
                           dtype='float64',
                           remove_exist=True)

    jobs = [("name%d" % i, np.arange(i * 10, i * 10 + 10)) for i in range(25)]
    all_data = {name: dat for name, dat in jobs}
    jobs = [
        (f, {name: dat for name, dat in j}) for j in np.array_split(jobs, 4)
    ]
    with Pool(2) as p:
      p.map(_fn_write2, jobs)
    f.flush()
    f.close()

    x = PointerArray(path)
    for name in x.indices:
      self.assertTrue(np.all(x[name] == all_data[name]))
    self.assertTrue(np.sum(x) == sum(np.sum(val) for val in all_data.values()))


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  unittest.main()
