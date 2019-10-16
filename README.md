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

About **535 times faster** than HDF5 data (using `h5py`) and **223 times faster** than normal `numpy.array`
[The detail benchmark code](https://github.com/trungnt13/bigarray/blob/master/benchmarks/mmap_vs_hdf5.py)
```
Array size: 1220.70 (MB)

Create HDF5   in: 0.0005580571014434099 s
Create Memmap in: 0.000615391880273819 s

Numpy save in: 0.5713834380730987 s
Writing data to HDF5  : 0.5530977640300989 s
Writing data to Memmap: 0.7038380969315767 s

Numpy saved size: 1220.70 (MB)
HDF5 saved size: 1220.71 (MB)
Mmap saved size: 1220.70 (MB)

Load Numpy array: 0.3723734531085938 s
Load HDF5 data  : 0.00041177100501954556 s
Load Memmap data: 0.00017150305211544037 s

Test correctness of stored data
Numpy : True
HDF5  : True
Memmap: True

Iterate Numpy data   : 0.00020254682749509811 s
Iterate HDF5 data    : 0.8945782391820103 s
Iterate Memmap data  : 0.0014937107916921377 s
Iterate Memmap (2nd) : 0.0011746759992092848 s

Numpy total time (open+iter): 0.3725759999360889 s
H5py  total time (open+iter): 0.8949900101870298 s
**Mmap  total time (open+iter): 0.001665213843807578 s**
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
