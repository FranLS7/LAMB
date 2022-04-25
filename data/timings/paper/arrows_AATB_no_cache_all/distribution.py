from statistics import median
import string
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

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
  data_dir_no_cache = "data/timings/paper/arrows_AATB_no_cache_all/"
  # data_dir = "../arrows_MCX_all/"
  # data_dir_no_cache = ""

  base_name = "anomaly"
  dim = "dim"

  n_anomalies = 1000 # number of initial anomalies from where dimensions are explored
  
  # This information depends on the expression
  ndim = 3
  n_algs = 5
  n_ops = 2
  n_samples = n_ops + 1
  jump = 10


  diff_prediction = np.array([])
  anomalies_measured = 0 # counter of the total number of anomalies measured
  anomalies_predicted = 0 # counter of the total number of anomalies without cache effects  

  conf_matrix = np.zeros((2, 2), dtype=np.int) # confusion matrix for the "classification"
  # In this confusion matrix rows indicate the original label whereas columns indicate the 
  # predicted label. 
  #              x0: true negatives (there is no anomaly and it's predicted)
  #   [ x0 x1 ]  x1: false positives (there is no anomaly but we predict to have one)  
  #   [ x2 x3 ]  x2: false negatives (there is an anomaly but we predict to have none)  
  #              x3: true positives (there is an anomaly and it's predicted)

  for anomaly_id in range(n_anomalies):
    for dim_id in range(ndim): 
      summary = pd.read_csv(data_dir + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_summary.csv")
      summary.sort_values(["d" + str(dim_id)], axis=0, inplace=True)
      summary = np.array(summary)
      ts_real = summary[:,-1]

      entries = np.nonzero(ts_real)[0] # nonzero returns a tuple with a np.array -> [0] to get the array
      ts_anomalies = summary[entries,-1]# The time scores measured 
      anomalies_measured += np.count_nonzero(ts_anomalies)

      summary_cache = pd.read_csv(data_dir_no_cache + base_name + str(anomaly_id) + "_dim" + str(dim_id) + "_summary.csv")
      summary_cache.sort_values(["d" + str(dim_id)], axis=0, inplace=True)
      summary_cache = np.array(summary_cache)
      ts_predicted = summary_cache[:,-1]

      ts_anomalies_predicted = summary_cache[entries,-1]
      anomalies_predicted += np.count_nonzero(ts_anomalies_predicted)

      file_difference = ts_anomalies - ts_anomalies_predicted
      diff_prediction = np.append(diff_prediction, file_difference)

      real_class = np.where(ts_real > 0.0, 1, 0)
      pred_class = np.where(ts_predicted > 0.0, 1, 0)
      conf_matrix += metrics.confusion_matrix(real_class, pred_class, labels=[0, 1])
      # metrics.classification_report(real_class, pred_class, labels=[0, 1])

  print(conf_matrix)

  fig, axis = plt.subplots()
  axis.grid(True)
  sns.distplot(diff_prediction)

  plt.show()




