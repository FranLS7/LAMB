from statistics import median
import string
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CPU_10FREQ = 1.5e9
FLOPS_DP = 16
N_CORES = 10

TPP = CPU_10FREQ * FLOPS_DP * N_CORES

if __name__ == "__main__":
  sns.set_style("dark")
  sns.set_context("poster")

  data_dir = "data/timings/paper/perf_profile/"
  # data_dir = ""

  filename = "gemm.csv"

  raw = pd.read_csv(data_dir + filename)
  data = np.array(raw)
  dims = data[:,0]
  
  times_gemm = data[:, 4:]
  median_gemm = np.median(times_gemm, axis=1)
  flops_gemm = data[:, 3]
  perf_gemm = flops_gemm / median_gemm
  eff_gemm = perf_gemm / TPP


  ###### data for SYRK ######
  filename = "syrk.csv"

  raw = pd.read_csv(data_dir + filename)
  data = np.array(raw)

  times_syrk = data[:, 3:]
  median_syrk = np.median(times_syrk, axis=1)
  flops_syrk = data[:, 2]
  perf_syrk = flops_syrk / median_syrk
  eff_syrk = perf_syrk / TPP


  ###### data for SYRK ######
  filename = "symm.csv"

  raw = pd.read_csv(data_dir + filename)
  data = np.array(raw)

  times_symm = data[:, 3:]
  median_symm = np.median(times_symm, axis=1)
  flops_symm = data[:, 2]
  perf_symm = flops_symm / median_symm
  eff_symm = perf_symm / TPP

  ###### Plotting the data ######
  fig, axis = plt.subplots(1)
  axis.plot(dims, eff_gemm, label="gemm")
  axis.plot(dims, eff_syrk, label="syrk")
  axis.plot(dims, eff_symm, label="symm")

  axis.grid(True)
  axis.legend()
  plt.xlabel('Size (m=k=n)')
  plt.ylabel('Efficiency')
  plt.show()