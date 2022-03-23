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
  # sns.set_context("talk")

  data_dir = "multi/timings/paper/volume_MCX/"
  # data_dir = ""
  filename = "vol_"
  # change the number of files when all regions are computed
  num_files = 100

  sizes = []
  ts = pd.DataFrame() # ts == time scores

  for i in range(num_files):
    read = pd.read_csv(data_dir + filename + str(i) + ".csv")
    sizes.append(read.shape[0]) # number of rows == number of anomalous instances
    ts = pd.concat([ts, read["time_score"]], axis=1)
    
  # sizes = np.array(sizes)
  # indices = np.flip(np.argsort(sizes))
  # sizes = sizes[indices]

  # data = np.array(ts, dtype=np.float)
  # data = data[:, indices]
  # median_ts = np.nanmean(data, axis=0)
  # median_ts = np.nan_to_num(median_ts)

  # fig = plt.figure()
  # plt.grid(True)
  # plt.stem(sizes, linefmt=':')

  # ax = fig.axes[0]
  # ax.set_ylabel("Anomalies in Region")
  # ax.set_yticks(np.arange(0, 1001, 200))

  # ax2 = ax.twinx()
  # ax2 = sns.boxplot(data=data, fliersize=0.0) #color="Set2") # palette="pink_r"
  # ax2.set_ylabel("Time Score")
  # ax2.set_xticks(np.arange(0, num_files, 5))
  # # ax2.set_yticks(np.linspace(0.1, 0.55, 10))
  
  # # norm = median_ts / max(median_ts)
  # norm_2 = colors.Normalize(vmin=0.10, vmax=max(median_ts + 0.05), clip=True)
  # mapper = cm.ScalarMappable(norm=norm_2, cmap=cm.get_cmap('YlOrRd'))
  # for i in range(0, 99):
  #   mybox = ax2.artists[i]
  #   mybox.set_facecolor(mapper.to_rgba(median_ts[i]))

  # plt.show()


  data = np.array(ts, dtype=np.float)
  median_ts = np.nanmedian(data, axis=0)
  median_ts = np.nan_to_num(median_ts)
  indices = np.flip(np.argsort(median_ts))
  median_ts = median_ts[indices]

  data = data[:, indices]
  sizes = np.array(sizes)
  sizes[indices]

  fig = plt.figure()
  plt.grid(True)
  ax = sns.boxplot(data=data, fliersize=0.0)
  ax.set_ylabel("Time Score")
  ax.set_xticks(np.arange(0, num_files, 5))


  norm_2 = colors.Normalize(vmin=0.10, vmax=max(median_ts) + 0.05, clip=True)
  mapper = cm.ScalarMappable(norm=norm_2, cmap=cm.get_cmap('YlOrRd'))
  for i in range(0, 99):
    mybox = ax.artists[i]
    mybox.set_facecolor(mapper.to_rgba(median_ts[i]))

  ax2 = ax.twinx()
  plt.stem(sizes, linefmt=':')
  ax2.set_ylabel("Anomalies in Region")
  ax2.set_yticks(np.arange(0, 1001, 200))
  plt.show()








  # data = pd.read_csv(data_dir + filename)

  # dims = data[["d0", "d1", "d2", "d3", "d4"]]
  # print(dims.shape)
  # kaka = np.unique(dims, axis=0)
  # print(kaka.shape)
