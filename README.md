# bigarray
Fast and scalable numpy array using Memory-mapped I/O

Stable build:
> pip install bigarray

Nightly build from github:
> pip install git+https://github.com/trungnt13/bigarray@master

---
## The three principles

* **Transparency**: everything is `numpy.array`, metadata and support for extra features (e.g. multiprocessing, indexing, etc) are subtly implemented in the _background_.
* **Pragmatism**: fast but easy, simplified A.P.I for common use cases
* **Focus**: _"Do one thing and do it well"_

## The benchmarks

[The detail benchmark code](https://github.com/trungnt13/bigarray/blob/master/benchmarks/mmap_vs_hdf5.py)
```
Create HDF5   in: 0.0006376712117344141 s
Create Memmap in: 0.0008840130176395178 s

Writing data to HDF5  : 0.273669223068282 s
Writing data to Memmap: 0.2863028401043266 s

Load HDF5 data  : 0.0006310669705271721 s
Load Memmap data: 0.00028447690419852734 s

Test correctness of stored data
HDF5  : True
Memmap: True

Iterate Numpy data   : 0.001672954997047782 s
Iterate HDF5 data    : 0.7986438490916044 s
Iterate Memmap data  : 0.011516761034727097 s
Iterate Memmap (2nd) : 0.011325995903462172 s
```

## Example

```python
from multiprocessing import Pool

import numpy as np

from bigarray import PointerArray, PointerArrayWriter

n = 80 * 10  # total number of samples
jobs = [(i, i + 10) for i in range(0, n // 10, 10)]
path = '/tmp/array'

# ====== Multiprocessing writing ====== #
writer = PointerArrayWriter(path, shape=(n,), dtype='int32', remove_exist=True)


def fn_write(job):
  start, end = job
  # it is crucial to write at different position for different process
  writer.write(
      {"name%i" % i: np.arange(i * 10, i * 10 + 10) for i in range(start, end)},
      start_position=start * 10)


# using 2 processes to generate and write data
with Pool(2) as p:
  p.map(fn_write, jobs)
writer.flush()
writer.close()

# ====== Multiprocessing reading ====== #
x = PointerArray(path)
print(x['name0'])
print(x['name66'])
print(x['name78'])

# normal indexing
for name, (s, e) in x.indices.items():
  data = x[s:e]
# fast indexing
for name in x.indices:
  data = x[name]


# multiprocess indexing
def fn_read(job):
  start, end = job
  total = 0
  for i in range(start, end):
    total += np.sum(x['name%d' % i])
  return total

# use multiprocessing to calculate the sum of all arrays
with Pool(2) as p:
  total_sum = sum(p.map(fn_read, jobs))
print(np.sum(x), total_sum)
```

Output:
```
[0 1 2 3 4 5 6 7 8 9]
[660 661 662 663 664 665 666 667 668 669]
[780 781 782 783 784 785 786 787 788 789]
319600 319600
```
