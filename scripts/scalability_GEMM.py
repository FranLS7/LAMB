import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

max_nthreads = 10
ndim = 3

SERVER_AVX512_FREQ = np.array([2e9, 2e9, 1.8e9, 1.8e9, 1.6e9, 1.6e9, 1.6e9, 1.6e9, 1.5e9, 1.5e9])
SERVER_DP = 16

tpp = np.arange(1, max_nthreads + 1, 1) * SERVER_AVX512_FREQ * SERVER_DP

data_dir = "../multi/timings/GEMM_scalability/"
file_base = "scalability_nth_"

def gemm_flops (dims):
    return (2 * dims[0] * dims[1] * dims[2])

times = []

for i in range (1, max_nthreads + 1):
    times.append (np.genfromtxt(data_dir + file_base + str(i) + ".csv", delimiter=',', skip_header=1))

sizes = times[0][:, 0:3]
flops = np.apply_along_axis(gemm_flops, 1, sizes)

times = [times[i][:, ndim + 1::] for i in range (len(times))]
median_times = [np.median(times[i], axis=1) for i in range(len(times))]


eff = [flops / median_times[i] / tpp[i] for i in range(len(median_times))]


plt.figure()
plt.xlabel('m (m=k=n)')
plt.ylabel('Efficiency [%]')
plt.title('GEMM Efficiency for Different #Cores')
plt.grid()

for i in range(len(median_times)):
    plt.plot(sizes[:,0], eff[i], label=str(i + 1) + " core(s)")
plt.legend()
plt.show()


scalability = [median_times[0] / median_times[i] / (i+1) for i in range(len(median_times))]

plt.figure()
plt.xlabel('m (m=k=n')
plt.ylabel('Scalability [%]')
plt.title('GEMM Scalability for Different #Cores')
plt.grid()

for i in range(len(median_times)):
    plt.plot(sizes[:,0], scalability[i], label=str(i+1) + " core(s)")
plt.legend()
plt.show()






