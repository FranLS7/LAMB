import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dir_sin = './MC3/timings/3D/'
filename = 'GEMM_m'

start = 50
stop = 1501
step = 50


points_k = np.arange(start, stop, step, dtype=np.double)
points_k = np.reshape(points_k, (points_k.shape[0], 1))

points_n = np.arange(start, stop, step, dtype=np.double)
points_n = np.reshape(points_n, (1, points_n.shape[0]))

points = int(stop / step)
gemm_cube = np.zeros((points, points, points), dtype=np.double)



for m in range(start, stop, step):
    fname = dir_sin + filename + str(m) + '.csv'
    slice_m = np.genfromtxt(fname, dtype=np.double, delimiter=', ')
    slice_m = slice_m[1:, 1:]
    
    flops = 2 * m * points_k * points_n
    perf_slice_m = flops / slice_m
        
    gemm_cube[:, :, int(m/step)-1] = perf_slice_m


m_arr = np.zeros((points ** 3), dtype=np.int)

for i in range(points):
    value = (i+1) * step
    m_arr[i * points ** 2 : (i+1) * points ** 2] = value


k_arr = np.zeros((points ** 3), dtype=np.int)

for i in range(points):
    value = step
    for j in range(points):
        k_arr[i * points ** 2 + j * points : i * points ** 2 + (j+1) * points] = value
        value = value + step


n_arr = np.zeros((points ** 3), dtype=np.int)
for i in range(points):
    for j in range(points):
        n_arr[i * points ** 2 + j * points : i * points ** 2 + (j+1) * points] = np.linspace(start, stop - 1, points)


fig = plt.figure(1)
plt.clf()

ax = fig.add_subplot (111, projection='3d')

img = ax.scatter (m_arr, n_arr, k_arr, c=gemm_cube, cmap='magma')
fig.colorbar (img)
plt.show()




