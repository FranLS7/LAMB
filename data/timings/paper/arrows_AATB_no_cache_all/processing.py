from statistics import median
import string
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm, colors

CPU_10FREQ = 1.5e9
FLOPS_DP = 16
N_CORES = 10

TPP = CPU_10FREQ * FLOPS_DP * N_CORES

def findDim (data, ndim):
  dims = data[0, :ndim] - data[1, :ndim]

  for i in range(ndim):
    if dims[i] != 0:
      return i


def findTransition(arr):
  idx = []

  for i in range(0, len(arr) - 1):
    if arr[i] == 0.0 and arr[i + 1] != 0.0:
      idx.append(i)
    elif arr[i] != 0.0 and arr[i + 1] == 0.0:
      idx.append(i)
  
  return idx


if __name__ == "__main__":
  sns.set_style("dark")
  # sns.set_context("talk")

  data_dir = "data/timings/paper/arrows_AATB_all/"
  data_dir_no_cache = "data/timings/paper/arrows_MCX_no_cache_all/"
  # data_dir = "../arrows_MCX_all/"
  # data_dir_no_cache = ""

  base_name = "anomaly"
  dim = "dim"
  anomaly_id = 77 # others generated: (35,0) (0,3) (98,4) (96,3) (95,4)
  dim_id = 1      # (79,1) (77,1)
  
  # All this information depends on the size of the matrix chains
  ndim = 3
  n_algs = 5
  n_ops = 2
  n_samples = n_ops + 1
  jump = 10
  # n_consecutive = 4

  data = []

  # Find the dimensions along which we move
  raw = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_alg0.csv")
  raw = np.array(raw)
  dim_move = findDim(raw, ndim)

  # Get the dimensions of the points explored
  raw = raw[raw[:, dim_move].argsort()]
  dims = raw[:, :ndim]

  # Read all the files and append them in a single list
  for i in range(n_algs):
    raw = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_alg" + str(i) + ".csv")    # load the data 
    raw.sort_values(["d" + str(dim_move)], axis=0, inplace=True) # sort by the dimension of interest
    arr = np.array(raw)[:, ndim:] # Convert to np.array and remove the dimensions 
    data.append(arr)              # only flops and times are remain there
  
  # Load the summary and convert it to numpy array
  summary = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_summary.csv")
  summary.sort_values(["d" + str(dim_move)], axis=0, inplace=True)
  summary = np.array(summary)

  # Compute the transition points
  transitions = findTransition(summary[:, -1])
  dim_transition = dims[transitions, dim_move]
  dim_transition = dim_transition + int(jump / 2)
  
  # Get flops and times
  flops = []
  times = []
  for alg in data:
    flops.append(alg[:, :n_samples])
    times.append(alg[:, n_samples:])

  min_times = summary[:, 6]
  min_flops = summary[:, 7]

  # Compute performance 
  perf = []
  for flop, time in zip(flops, times):
    perf.append(flop / time)

  # Compute efficiency
  eff = []
  for p in perf:
    eff.append(p / TPP)

  # ============================== WITHOUT CACHE ==============================
  # Read all the files without cache and append them in a single list
  data_cache = []
  for i in range(n_algs):
    raw_cache = pd.read_csv(data_dir_no_cache + base_name + str(anomaly_id) + "_dim" + str(dim_id) 
      + "_alg" + str(i) + ".csv")    # load the data 
    raw_cache.sort_values(["d" + str(dim_move)], axis=0, inplace=True) # sort by the dimension of interest
    arr = np.array(raw_cache)[:, ndim:] # Convert to np.array and remove the dimensions 
    data_cache.append(arr)              # only flops and times are remain there

  # Load the summary and convert it to numpy array
  summary_cache = pd.read_csv(data_dir_no_cache + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_summary.csv")
  summary_cache.sort_values(["d" + str(dim_move)], axis=0, inplace=True)
  summary_cache = np.array(summary_cache)

  # Get times without cache
  times_cache = []
  for alg in data_cache:
    times_cache.append(alg[:, n_samples:])
  
  min_times_cache = summary_cache[:, 6]

  # Compute performance 
  perf_cache = []
  for flop, time in zip(flops, times_cache):
    perf_cache.append(flop / time)

  # Compute efficiency
  eff_cache = []
  for p in perf_cache:
    eff_cache.append(p / TPP)


  # ================================ PLOTTING =================================

  # Plotting the results
  bl = 0.0 # bottom line
  # fig, axes = plt.subplots(6, 2) # To print the FLOP counts next to the performances
  # For the FLOP count change axes[i] to axes[i, 0]

  fig, axes = plt.subplots(n_algs)
  for i in range(n_algs):
    axes[i].plot(dims[:, dim_move], eff[i][:, -1], label='Total', linewidth=3.0)
    axes[i].plot(dims[:, dim_move], eff_cache[i][:, -1], label='T. No Cache', linewidth=2.5)
    # line.set_label('total execution')

    # To plot the individual GEMMs in each algorithm
    # for op in range(n_ops):
    #   axes[i].plot(dims[:, dim_move], eff[i][:, op])

    axes[i].grid(True)
    axes[i].set_ylim(bl, 1.0)
    for change in dim_transition:
      axes[i].axvline(x=change, linestyle='--', color='red')

    axes[i].fill_between(dims[:, dim_move], bl, 1, where=min_times == i, 
                         facecolor='green', alpha=0.2)
    axes[i].fill_between(dims[:, dim_move], bl, 1, where=min_flops == i,
                         facecolor='red', alpha=0.2)
  
  axes[0].legend(["Total", "T. No Cache"])#, "First", "Second", "Third"])

  fig2, axes2 = plt.subplots()
  axes2.grid(True)
  axes2.plot(dims[:, dim_move], summary[:, -1], label='Total')
  axes2.plot(dims[:, dim_move], summary_cache[:, -1], label="T. No Cache")

  # To print the FLOP counts next to the performances           
  # for i in range(n_algs):
  #   axes[i, 1].plot(dims[:, dim_move], flops[i][:, -1])
  #   axes[i, 1].grid(True)
  #   for change in dim_transition:
  #     axes[i, 1].axvline(x=change, linestyle='--', color='red')
  #   axes[i, 1].set_ylim((1e8, 2e9))

  #   lims = axes[i, 1].get_ylim()
  #   axes[i, 1].fill_between(dims[:, dim_move], lims[0], lims[1], where=min_times == i,
  #                           facecolor='green', alpha=0.2)
  #   axes[i, 1].fill_between(dims[:, dim_move], lims[0], lims[1], where=min_flops == i,
  #                           facecolor='red', alpha=0.2)

  plt.show()