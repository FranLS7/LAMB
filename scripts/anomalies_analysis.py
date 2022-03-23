import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


# def main():
data_dir = '../multi/timings/MC3_mt/'
filename = 'MC4_replicate_an.csv'
data = np.genfromtxt (data_dir + filename, delimiter=',', skip_header=1)

gemm1 = data[0][6::4]
gemm2 = data[0][7::4]
gemm3 = data[0][8::4]
p1 = data[0][9::4]

plt.figure()
hist_gemm1 = plt.hist (gemm2, bins='auto')
hist_gemmx = plt.hist (data[1][7::4], bins='auto')
plt.grid()
plt.figure()
hist_gemm2 = plt.hist (gemm3, bins='auto')
hist_gemmxx = plt.hist (data[1][8::4], bins='auto')
plt.grid()
plt.figure()
hist_p1 = plt.hist (p1, bins='auto')
hist_gemmxxx = plt.hist (data[1][9::4], bins='auto')
plt.grid()

