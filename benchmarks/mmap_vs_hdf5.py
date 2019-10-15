from __future__ import absolute_import, division, print_function

import os
import timeit

import h5py
import numpy as np

from bigarray import MmapArray, MmapArrayWriter

hdf5_path = '/tmp/tmp.hdf5'
mmap_path = '/tmp/tmp.mmap'
numpy_path = '/tmp/tmp.array'

N = 50000
X = np.random.rand(N, 25, 128).astype('float64')
print("Array size: %.2f (MB)\n" %
      (np.prod(X.shape) * X.dtype.itemsize / 1024 / 1024))

# ====== test created dataset ====== #
start = timeit.default_timer()
hdf5 = h5py.File(hdf5_path, 'w')
print('Create HDF5   in:', timeit.default_timer() - start, 's')

start = timeit.default_timer()
mmap = MmapArrayWriter(mmap_path,
                       dtype='float64',
                       shape=(0,) + X.shape[1:],
                       remove_exist=True)
print('Create Memmap in:', timeit.default_timer() - start, 's')

# ====== writing ====== #
print()

start = timeit.default_timer()
with open(numpy_path, 'wb') as f:
  np.save(f, X)
print('Numpy save in:', timeit.default_timer() - start, 's')

start = timeit.default_timer()
hdf5['X'] = X
print('Writing data to HDF5  :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
mmap.write(X)
print('Writing data to Memmap:', timeit.default_timer() - start, 's')

hdf5.flush()
hdf5.close()
mmap.flush()
mmap.close()

# ====== check file size ====== #
print()
print("Numpy saved size: %.2f (MB)" %
      (os.stat(numpy_path).st_size / 1024 / 1024))
print("HDF5 saved size: %.2f (MB)" % (os.stat(hdf5_path).st_size / 1024 / 1024))
print("Mmap saved size: %.2f (MB)" % (os.stat(mmap_path).st_size / 1024 / 1024))

# ====== reading ====== #
print()

start = timeit.default_timer()
with open(numpy_path, 'rb') as f:
  y = np.load(f)
print('Load Numpy array:', timeit.default_timer() - start, 's')

start = timeit.default_timer()
hdf5 = h5py.File(hdf5_path, 'r')
print('Load HDF5 data  :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
mmap = MmapArray(mmap_path)
print('Load Memmap data:', timeit.default_timer() - start, 's')

print()
print('Test correctness of stored data')
print('Numpy :', np.all(y == X))
print('HDF5  :', np.all(hdf5['X'][:] == X))
print('Memmap:', np.all(mmap[:] == X))

# ====== iterating over dataset ====== #
print()
start = timeit.default_timer()
for epoch in range(0, 3):
  for i in range(0, N, 256):
    x = X[i:i + 256]
print('Iterate Numpy data   :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
dat = hdf5['X']
for epoch in range(0, 3):
  for i in range(0, N, 256):
    x = dat[i:i + 256]
print('Iterate HDF5 data    :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
for epoch in range(0, 3):
  for i in range(0, N, 256):
    x = mmap[i:i + 256]
print('Iterate Memmap data  :', timeit.default_timer() - start, 's')

start = timeit.default_timer()
for epoch in range(0, 3):
  for i in range(0, N, 256):
    x = mmap[i:i + 256]
print('Iterate Memmap (2nd) :', timeit.default_timer() - start, 's')

# ===========================================================================
# Clean-up
# ===========================================================================
if os.path.exists(numpy_path):
  os.remove(numpy_path)
if os.path.exists(hdf5_path):
  os.remove(hdf5_path)
if os.path.exists(mmap_path):
  os.remove(mmap_path)
