import numpy as np
import sys
import time

from cube_lib import compute_cube

dir_out = '../cube/timings/'

min_size = np.array ([20, 20, 20], dtype=np.int32)
max_size = np.array ([2000, 2000, 2000], dtype=np.int32)
npoints = np.array ([40, 40, 40], dtype=np.int32)
nsamples = 10
nthreads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
output_files = ['cube_' + str(x) + 'cores.csv' for x in nthreads]
print(output_files)

for i in range(len(nthreads)):
    # print(nthreads[i])
    compute_cube(min_size, max_size, npoints, nsamples, nthreads[i], dir_out + output_files[i])