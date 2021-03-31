import numpy as np
import sys
import time

from cube_lib import compute_cube

dir_out = '../cube/timings/'

min_size = np.array ([700, 700, 700], dtype=np.int32)
max_size = np.array ([1000, 1000, 1000], dtype=np.int32)
npoints = np.array ([5, 5, 5], dtype=np.int32)
nsamples = 10
nthreads = [2]
output_files = ["caca.csv"]

for i in range(len(nthreads)):
    # print(nthreads[i])
    compute_cube(min_size, max_size, npoints, nsamples, nthreads[i], dir_out + output_files[i])