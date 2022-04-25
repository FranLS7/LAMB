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

  data_dir = "data/timings/paper/arrows_MCX_all/"
  # data_dir = ""
  base_name = "anomaly"
  dim = "dim"
  anomaly_id = 73 # (35,0)
  dim_id = 4
  
  # All this information depends on the size of the matrix chains
  ndim = 5
  n_algs = 6
  n_ops = 3
  n_samples = n_ops + 1
  jump = 10
  # n_consecutive = 4

  data = []

  # Find the dimensions along which we move
  raw = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_alg0.csv")
  raw = np.array(raw)
  dim_move_0 = findDim(raw, ndim)

  # Get the dimensions of the points explored
  raw = raw[raw[:, dim_move_0].argsort()]
  dims_0 = raw[:, :ndim]

  # Read all the files and append them in a single list
  for i in range(n_algs):
    raw = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_alg" + str(i) + ".csv")    # load the data 
    raw.sort_values(["d" + str(dim_move_0)], axis=0, inplace=True) # sort by the dimension of interest
    arr = np.array(raw)[:, ndim:] # Convert to np.array and remove the dimensions 
    data.append(arr)              # only flops and times are remain there
  
  # Load the summary and convert it to numpy array
  summary = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_summary.csv")
  summary.sort_values(["d" + str(dim_move_0)], axis=0, inplace=True)
  summary = np.array(summary)

  # Compute the transition points
  transitions = findTransition(summary[:, -1])
  dim_transition = dims_0[transitions, dim_move_0]
  dim_transition = dim_transition + int(jump / 2)
  
  # Get flops and times
  flops = []
  times = []
  for alg in data:
    flops.append(alg[:, :n_samples])
    times.append(alg[:, n_samples:])
  
  flops_alg_0 = np.zeros((dims_0.shape[0], n_algs))
  for i in range(len(flops)):
    flops_alg_0[:,i] = flops[i][:,-1]
  flops_alg_0 /= np.max(flops_alg_0)

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

  # Plotting the results
  bl = 0.35 # bottom line
  # fig, axes = plt.subplots(6, 2) # To print the FLOP counts next to the performances
  # For the FLOP count change axes[i] to axes[i, 0]

  fig, axes = plt.subplots(n_algs, 2)
  for i in range(n_algs):
    axes[i,0].plot(dims_0[:, dim_move_0], eff[i][:, -1], label='Total', linewidth=3.0)

    # To plot the individual GEMMs in each algorithm
    for op in range(n_ops):
      axes[i,0].plot(dims_0[:, dim_move_0], eff[i][:, op])

    if (i != n_algs-1):
      axes[i,0].xaxis.set_ticklabels([])

    axes[i,0].grid(True)
    axes[i,0].set_ylim(bl, 1.0)
    for j in range(2):
    # for change in dim_transition:
      axes[i,0].axvline(x=dim_transition[j], linestyle='--', color='red')

    if (i == 4):
      axes[i,0].fill_between(dims_0[:, dim_move_0], bl, 1, where=min_times == i, 
                          facecolor='green', alpha=0.2)
      axes[i,0].fill_between(dims_0[:, dim_move_0], bl, 1, where=min_flops == 1,
                          facecolor='red', alpha=0.2)
    
    else:
      axes[i,0].fill_between(dims_0[:, dim_move_0], bl, 1, where=min_times == i, 
                          facecolor='green', alpha=0.2)
      axes[i,0].fill_between(dims_0[:, dim_move_0], bl, 1, where=min_flops == i,
                          facecolor='red', alpha=0.2)
    
  axes[0,0].legend(["Total", "First", "Second", "Third"])
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Dimension size')
  plt.ylabel('Efficiency')
  axes[5,0].axhline(y=bl, xmin=0.22 , xmax=0.65 , color='black', linewidth=7.0)



  # ==================================== SECOND DIMENSION SWEEP ====================================

  anomaly_id = 66
  dim_id = 3

  data = []

  # Find the dimensions along which we move
  raw = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_alg0.csv")
  raw = np.array(raw)
  dim_move = findDim(raw, ndim)

  # Get the dimensions of the points explored
  raw = raw[raw[:, dim_move].argsort()]
  dims_1 = raw[:, :ndim]

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
  dim_transition = dims_1[transitions, dim_move]
  dim_transition = dim_transition + int(jump / 2)
  
  # Get flops and times
  flops = []
  times = []
  for alg in data:
    flops.append(alg[:, :n_samples])
    times.append(alg[:, n_samples:])
  
  flops_alg_1 = np.zeros((dims_1.shape[0], n_algs))
  for i in range(len(flops)):
    flops_alg_1[:,i] = flops[i][:,-1]
  flops_alg_1 /= np.max(flops_alg_1)

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

  # bl = 0.0
  for i in range(n_algs):
    axes[i,1].plot(dims_1[:, dim_move], eff[i][:, -1], label='Total', linewidth=3.0)

    # To plot the individual GEMMs in each algorithm
    for op in range(n_ops):
      axes[i,1].plot(dims_1[:, dim_move], eff[i][:, op])

    if (i != n_algs-1):
      axes[i,1].xaxis.set_ticklabels([])
    
    axes[i,1].yaxis.set_ticklabels([])

    axes[i,1].grid(True)
    axes[i,1].set_ylim(bl, 1.0)
    for change in dim_transition:
      axes[i,1].axvline(x=change, linestyle='--', color='red')

    axes[i,1].fill_between(dims_1[:, dim_move], bl, 1, where=min_times == i, 
                         facecolor='green', alpha=0.2)
    axes[i,1].fill_between(dims_1[:, dim_move], bl, 1, where=min_flops == i,
                         facecolor='red', alpha=0.2)
  
  axes[5,1].axhline(y=bl, xmin=0.13 , xmax=0.93 , color='black', linewidth=7.0)

  fig, axes = plt.subplots(1, 2)
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Dimension size')
  plt.ylabel('Normalised FLOPs')
  
  for i in range(n_algs):
    axes[0].grid(True)
    axes[0].plot(dims_0[:, dim_move_0], flops_alg_0[:,i])
  for i in range(n_algs):
    axes[1].grid(True)
    axes[1].plot(dims_1[:, dim_move], flops_alg_1[:,i])#, color='black')
  axes[0].legend(["Alg1", "Alg2", "Alg3", "Alg4", "Alg5", "Alg6"])

  plt.show()