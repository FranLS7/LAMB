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


def findTransitionUp(arr):
  idx = []

  for i in range(0, len(arr) - 1):
    if arr[i] == 0.0 and arr[i + 1] != 0.0:
      idx.append(i)
  
  return idx


if __name__ == "__main__":
  sns.set_style("dark")
  # sns.set_context("talk")

  data_dir = "data/timings/paper/arrows_AATB_all/"
  # data_dir = ""
  base_name = "anomaly"
  dim = "dim"
  anomaly_id = 10 # 302
  dim_id = 0
  
  # All this information depends on the size of the matrix chains
  ndim = 3
  n_algs = 5
  n_ops = 2
  n_samples = n_ops + 1
  jump = 10
  # n_consecutive = 4

  legends = [["Total", "1.syrk", "2.symm"],
            ["Total", "1.syrk", "2.gemm"],
            ["Total", "1.gemm", "2.symm"],
            ["Total", "1.gemm", "2.gemm"],
            ["Total", "1.gemm", "2.gemm"]]

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

  min_times = summary[:, 4]
  min_flops = summary[:, 5]

  # Compute performance 
  perf = []
  for flop, time in zip(flops, times):
    perf.append(flop / time)  

  # Compute efficiency
  eff = []
  for p in perf:
    eff.append(p / TPP)

  # Find min in time
  total_times = times[0][:,-1]
  for i in range(1, len(times)):
    total_times = np.column_stack((total_times, times[i][:,-1]))
  alg_min_time = np.argmin(total_times, axis=1)

  # Plotting the results
  bl = 0.0 # bottom line
  # fig, axes = plt.subplots(6, 2) # To print the FLOP counts next to the performances
  # For the FLOP count change axes[i] to axes[i, 0]

  fig, axes = plt.subplots(n_algs, 3)
  for i in range(n_algs):
    axes[i,0].plot(dims[:, dim_move], eff[i][:, -1], label='Total', linewidth=3.0)

    # To plot the individual GEMMs in each algorithm
    for op in range(n_ops):
      axes[i,0].plot(dims[:, dim_move], eff[i][:, op])

    if (i != n_algs-1):
      axes[i,0].xaxis.set_ticklabels([])

    axes[i,0].grid(True)
    axes[i,0].set_ylim(bl, 1.0)
    for change in dim_transition:
      axes[i,0].axvline(x=change, linestyle='--', color='red')

    # if (i == 4):
    #   axes[i,0].fill_between(dims[:, dim_move], bl, 1, where=min_times == i, 
    #                       facecolor='green', alpha=0.2)
    #   axes[i,0].fill_between(dims[:, dim_move], bl, 1, where=min_flops == 1,
    #                       facecolor='red', alpha=0.2)
    
    # else:
    draw_here = np.where(alg_min_time == i, 1, 0)
    draw_transitions = findTransitionUp(draw_here)
    for trans in draw_transitions:
      draw_here[trans] = 1
      draw_here[trans + 1] = 1 

    axes[i,0].fill_between(dims[:, dim_move], bl, 1, where=draw_here, 
                        facecolor='green', alpha=0.2)


    if (i < 2):
      draw_here = np.where(min_flops < 2, 1, 0)
    else:
      draw_here = np.where(min_flops == i, 1, 0)

    draw_transitions = findTransition(draw_here)
    for trans in draw_transitions:
      draw_here[trans] = 1
      draw_here[trans + 1] = 1 

    axes[i,0].fill_between(dims[:, dim_move], bl, 1, where=draw_here,
                        facecolor='red', alpha=0.2)
                        
  axes[4,0].axhline(y=bl, xmin=0.05 , xmax=0.80 , color='black', linewidth=7.0)  
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Dimension size', fontsize=13)
  plt.ylabel('Efficiency', fontsize=13)



  # ==================================== SECOND DIMENSION SWEEP ====================================

  anomaly_id = 13 # 25, 11
  dim_id = 1

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

  min_times = summary[:, 4]
  min_flops = summary[:, 5]

  # Compute performance 
  perf = []
  for flop, time in zip(flops, times):
    perf.append(flop / time)  

  # Compute efficiency
  eff = []
  for p in perf:
    eff.append(p / TPP)
  
  # Find min in time
  total_times = times[0][:,-1]
  for i in range(1, len(times)):
    total_times = np.column_stack((total_times, times[i][:,-1]))
  alg_min_time = np.argmin(total_times, axis=1)

  # bl = 0.0
  for i in range(n_algs):
    axes[i,1].plot(dims[:, dim_move], eff[i][:, -1], label='Total', linewidth=3.0)

    # To plot the individual GEMMs in each algorithm
    for op in range(n_ops):
      axes[i,1].plot(dims[:, dim_move], eff[i][:, op])

    if (i != n_algs-1):
      axes[i,1].xaxis.set_ticklabels([])
    
    axes[i,1].yaxis.set_ticklabels([])

    axes[i,1].grid(True)
    axes[i,1].set_ylim(bl, 1.0)
    for change in dim_transition:
      axes[i,1].axvline(x=change, linestyle='--', color='red')

    draw_here = np.where(alg_min_time == i, 1, 0)
    draw_transitions = findTransitionUp(draw_here)
    for trans in draw_transitions:
      draw_here[trans] = 1
      draw_here[trans + 1] = 1 

    axes[i,1].fill_between(dims[:, dim_move], bl, 1, where=draw_here, 
                        facecolor='green', alpha=0.2)
    
    if (i < 2):
      draw_here = np.where(min_flops < 2, 1, 0)
    else:
      draw_here = np.where(min_flops == i, 1, 0)
    draw_transitions = findTransition(draw_here)
    for trans in draw_transitions:
      draw_here[trans] = 1
      draw_here[trans + 1] = 1 

    axes[i,1].fill_between(dims[:, dim_move], bl, 1, where=draw_here,
                        facecolor='red', alpha=0.2)

  axes[4,1].axhline(y=bl, xmin=0.1 , xmax=0.93 , color='black', linewidth=7.0)

  # ==================================== THIRD DIMENSION SWEEP ====================================
  anomaly_id = 0
  dim_id = 2

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

  min_times = summary[:, 4]
  min_flops = summary[:, 5]

  # Compute performance 
  perf = []
  for flop, time in zip(flops, times):
    perf.append(flop / time)  

  # Compute efficiency
  eff = []
  for p in perf:
    eff.append(p / TPP)
  
  # Find min in time
  total_times = times[0][:,-1]
  for i in range(1, len(times)):
    total_times = np.column_stack((total_times, times[i][:,-1]))
  alg_min_time = np.argmin(total_times, axis=1)

  # bl = 0.0
  for i in range(n_algs):
    axes[i,2].plot(dims[:, dim_move], eff[i][:, -1], label='Total', linewidth=3.0)

    # To plot the individual GEMMs in each algorithm
    for op in range(n_ops):
      axes[i,2].plot(dims[:, dim_move], eff[i][:, op])

    if (i != n_algs-1):
      axes[i,2].xaxis.set_ticklabels([])
    
    axes[i,2].yaxis.set_ticklabels([])

    axes[i,2].grid(True)
    axes[i,2].set_ylim(bl, 1.0)
    for change in dim_transition:
      axes[i,2].axvline(x=change, linestyle='--', color='red')

    draw_here = np.where(alg_min_time == i, 1, 0)
    draw_transitions = findTransitionUp(draw_here)
    for trans in draw_transitions:
      draw_here[trans] = 1
      draw_here[trans + 1] = 1 

    axes[i,2].fill_between(dims[:, dim_move], bl, 1, where=draw_here, 
                        facecolor='green', alpha=0.2)
    
    if (i < 2):
      draw_here = np.where(min_flops < 2, 1, 0)
    else:
      draw_here = np.where(min_flops == i, 1, 0)
    draw_transitions = findTransition(draw_here)
    for trans in draw_transitions:
      draw_here[trans] = 1
      draw_here[trans + 1] = 1 

    axes[i,2].fill_between(dims[:, dim_move], bl, 1, where=draw_here,
                        facecolor='red', alpha=0.2)
  
    leg = axes[i,2].legend(legends[i])
    if (i == 1):
      bb = leg.get_bbox_to_anchor().inverse_transformed(axes[i,2].transAxes)
      bb.x1 -= 0.2
      leg.set_bbox_to_anchor(bb, transform=axes[i,2].transAxes)



    # axes[i,1].fill_between(dims[:, dim_move], bl, 1, where=min_times == i, 
    #                      facecolor='green', alpha=0.2)
    # axes[i,1].fill_between(dims[:, dim_move], bl, 1, where=min_flops == i,
    #                      facecolor='red', alpha=0.2)

  # axes[0,2].legend(["Total", "First", "Second"])

  axes[4,2].axhline(y=bl, xmin=0.1 , xmax=0.93 , color='black', linewidth=7.0)
  plt.show()