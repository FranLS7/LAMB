from statistics import median
import string
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm, colors

if __name__ == "__main__":
  sns.set_style("dark")
  sns.set_context("poster")

  data_dir = "data/timings/paper/arrows_AATB_all/"
  # data_dir = ""
  filename = "anomaly"
  # change the number of files when all regions are computed
  num_anomalies = 1000
  num_dims = 3
  step_size = 10
  
  sizes = np.zeros((num_anomalies, num_dims), dtype=np.int)
  ts = pd.DataFrame() # ts == time scores

  for dim in range(num_dims):
    for anomaly in range(num_anomalies):
      read = pd.read_csv(data_dir + filename + str(anomaly) + "_dim" + str(dim) + "_summary.csv")
      time_score = np.array(read["time_score"])
      sizes[anomaly, dim] = np.count_nonzero(time_score)

  sizes = np.sort(sizes, axis=0)
  sizes *= step_size

  fig, axes = plt.subplots(1, num_dims)
  for i in range(num_dims):
    axes[i].grid(True)

    axes[i].plot(sizes[:,i])
    axes[i].set_ylim([0, 1200])
    
    l1 = axes[i].lines[0]
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    axes[i].fill_between(x1, y1, color="blue", alpha=0.5)

    if (i != 0):
      axes[i].yaxis.set_ticklabels([])

  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Sorted randomly found anomalies', labelpad=10)
  plt.ylabel('Region thickness', labelpad=20)

  plt.show()