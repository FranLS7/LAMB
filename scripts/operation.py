import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

TPP_SERVER = 240e9

def gemm_flops (sizes):
    return (2 * sizes[0] * sizes[1] * sizes[2])

def symm_flops (sizes):
    return (2 * sizes[0] * sizes[0] * sizes[1])

def syrk_flops (sizes):
    return (sizes[0] * (sizes[0] + 1) * sizes[1])

def syr2k_flops (sizes):
    return (2 * sizes[0] * sizes[0] * sizes[1])

def trmm_flops (sizes):
    return (sizes[0] * sizes[0] * sizes[1])

def trsm_flops (sizes):
    return (sizes[0] * sizes[0] * sizes[1])

data_dir = "../kernels/timings/"
data_file = "syr2k_timings.csv"

class operation():
    def __init__(self, name_op, filename, ndim, func):
        self.name_op = name_op
        self.filename = filename
        self.ndim = ndim
        self.__loadData(filename)
        self.sizes = self.data[:, :ndim]
        self.times = self.data[:, ndim:]
        self.__getMedian()
        self.__getFlops(func)
        self.medianPerf = self.flops / self.medianTimes

    def __loadData (self, filename):
        self.data = np.genfromtxt (self.filename, delimiter=',', skip_header=1)

    def __getMedian (self):
        self.medianTimes = np.median (self.times, axis=1)

    def __getFlops (self, func):
        self.flops = np.apply_along_axis (func1d=func, axis=1, arr=self.sizes)

    def plotDiagonal (self, mode, tpp):
        plt.grid(True)
        # GET PROPER DIAGONAL

        if self.ndim == 2:
            plt.xlabel ("size: m=n")
        elif self.ndim == 3:
            plt.xlabel ("size: m=k=n")

        if mode=='time':
            plt.title (self.name_op + " Execution Time")
            plt.ylabel ("Time [s]")
            plt.plot (self.sizes[:,0], self.medianTimes, label=self.name_op + " " + mode)

        elif mode=='perf':
            plt.title (self.name_op + " Performance")
            plt.ylabel ("Performance [FLOPS]")
            plt.ylim (0, tpp)
            plt.plot (self.sizes[:,0], self.medianPerf, label=self.name_op + " " + mode)

        elif mode=='eff':
            plt.title (self.name_op + " Efficiency")
            plt.ylabel ("Efficiency [%]")
            plt.ylim (0, 1.0)
            plt.plot (self.sizes[:,0], self.medianPerf / tpp, label=self.name_op + " " + mode)

        else:
            print("Error: available mode types are 'time' 'perf' and 'eff'")





# TESTING

op = operation("SYR2K", data_dir + data_file, 2, syr2k_flops)
plt.figure
op.plotDiagonal('eff', 240e9)

op2 = operation("SYRK", data_dir + "syrk_timings.csv", 2, syrk_flops)
op2.plotDiagonal('eff', 240e9)

op3 = operation("SYMM", data_dir + "symm_timings.csv", 2, symm_flops)
op3.plotDiagonal('eff', 240e9)

op4 = operation("GEMM", data_dir + "gemm_timings.csv", 3, gemm_flops)
op4.plotDiagonal('eff', 240e9)

op5 = operation("TRMM", data_dir + "trmm_timings.csv", 2, trmm_flops)
op5.plotDiagonal('eff', 240e9)

op6 = operation("TRSM", data_dir + "trsm_timings.csv", 2, trsm_flops)
op6.plotDiagonal('eff', 240e9)

plt.legend()
plt.show()