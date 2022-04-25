import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# from matplotlib import rc

if __name__ == "__main__":
  sns.set_style("dark")
  sns.set_context('poster')

  # matplotlib.rcParams.update({'font.size': 52})
  # sns.set(font_scale = 12)
  # rc('font',**{'family':'serif','serif':['Palatino']})

  data_dir = "data/timings/paper/anomalies_MCX/"
  filename = "summary.csv"

  data = pd.read_csv(data_dir + filename)

  time_min = np.array(data["alg_i"])
  flop_min = np.array(data["alg_j"])
  time_score = np.array(data["time_score"])
  flop_score = np.array(data["flops_score"])

  plt.figure()
  plt.title("Distribution Alg Min Time")
  plt.xlabel("alg")
  plt.ylabel("num ocurrence")
  plt.hist(time_min, bins=5)

  plt.figure()
  plt.title("Distribution Alg Min FLOPs")
  plt.ylabel("num ocurrence")
  plt.xlabel("alg")
  plt.hist(flop_min, bins=5)

  plt.figure()
  plt.title("Distribution Time Score")
  plt.ylabel("num ocurrence")
  plt.xlabel("anomaly")
  plt.hist(time_score)  

  plt.figure()
  plt.title("Distribution Time Score")
  plt.ylabel("num ocurrence")
  plt.xlabel("time score")
  plt.grid(True)
  sns.histplot(x=time_score, palette="inferno_r")
  # plt.savefig("fig.pdf", format="pdf")

  fig, ax = plt.subplots()
  # plt.title("Time Score vs FLOP Score")
  plt.ylabel("Time score")
  plt.ylim(bottom=0.0, top=0.41)
  plt.xlabel("FLOP score")
  plt.xlim(left=0.0, right=0.51)
  plt.grid(True)
  distance = np.sqrt(time_score ** 2 + flop_score ** 2)
  sns.scatterplot(x=flop_score, y=time_score, hue=distance)
  ax.get_legend().remove()
  # plt.legend()
  

  plt.figure()
  data["time_score"].plot.density()
  data["flops_score"].plot.density()
  plt.grid(True)



  plt.show()
