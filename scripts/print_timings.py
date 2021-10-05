import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

data_dir = '../multi/timings/check_timings/'
filename = 'checking_CF.csv'
data = np.genfromtxt (data_dir + filename, delimiter=',', skip_header=1)
ndim = 3 


times = data[:, ndim + 1:]

op1 = times[0]
op2 = times[1]
op3 = times[2]

plt.plot (op1, label='CF + [GEMM2]')
plt.plot (op2, label='CF + same-GEMM + [GEMM2]')
plt.plot (op3, label='CF + GEMM1 + [GEMM2]')
plt.legend()
plt.grid()
plt.ylim(0)
plt.xlabel('samples')
plt.ylabel('execution time [s]')
plt.title ('Comparing chains of operations')

plt.figure()
for i in range(3, times.shape[0]):
    plt.plot (times[i], label='CF + GEMM(100 * ' +  str(i - 2) + ') + [GEMM2]')
plt.legend()
plt.grid()
plt.ylim(0)
plt.xlabel('samples')
plt.ylabel('execution time [s]')
plt.title ('Influence of rGEMM')

median_values = np.median (times, axis=1)
plt.figure()
plt.plot(median_values[3:])
# plt.ylim(0, 0.004)
plt.grid()
plt.xlabel ('Size rGEMM = 100 * x+1')
plt.ylabel ('Median execution time GEMM2 [s]')
plt.title ('Influence of rGEMM on GEMM2 (m=k=n=500)')

# gemm1 = data[0][6::4]
# gemm2 = data[0][7::4]
# gemm3 = data[0][8::4]
# p1 = data[0][9::4]

# plt.figure()
# hist_gemm1 = plt.hist (gemm2, bins='auto')
# hist_gemmx = plt.hist (data[1][7::4], bins='auto')
# plt.grid()
# plt.figure()
# hist_gemm2 = plt.hist (gemm3, bins='auto')
# hist_gemmxx = plt.hist (data[1][8::4], bins='auto')
# plt.grid()
# plt.figure()
# hist_p1 = plt.hist (p1, bins='auto')
# hist_gemmxxx = plt.hist (data[1][9::4], bins='auto')
# plt.grid()
