import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

TPP_SERVER = 240e9

def symm_flops (sizes):
    return (2 * sizes[0] * sizes[0] * sizes[1])

data_dir = "../multi/timings/symm/"
data_file = "symm_timings.csv"

data = np.genfromtxt (data_dir + data_file, delimiter=',', skip_header=1)
sizes = data[:,:2]
flops = np.apply_along_axis (func1d=symm_flops, axis=1, arr=sizes)

times = data[:, 2:]
median_times = np.median (times, axis=1)

median_eff = flops / median_times / TPP_SERVER

plt.figure();
plt.title ("SYMM Efficiency")
plt.xlabel ("size: m=n")
plt.ylabel ("Efficiency [%]")
plt.ylim(0, 1.0)
plt.grid(True)
plt.plot (sizes[:,1], median_eff)

