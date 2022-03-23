import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CPU_10FREQ = 1.5e9
FLOPS_DP = 16
N_CORES = 10

TPP = CPU_10FREQ * FLOPS_DP * N_CORES


if __name__ == "__main__":
    data_dir = "multi/timings/MCX/performance_profile/"
    filename = "gemm_1000.csv"

    x_dim = "d0"
    y_dim = "d2"

    data = pd.read_csv(data_dir + filename)
    
    proc_data = data.to_numpy()
    median = np.median(proc_data[:, 4:], axis=1)
    flops = proc_data [:, 3]
    performance = flops / median
    performance = np.reshape (performance, (len(performance), 1))
    efficiency = performance / TPP # change this only for SYMM it's * 2

    efficiency = np.hstack([proc_data[:,:3], efficiency])
    
    proc_data = pd.DataFrame (efficiency, columns=['d0', 'd1', 'd2', 'efficiency'])
    
    perf = proc_data.pivot(index=x_dim, columns=y_dim, values="efficiency")
    # anomalies = data.pivot(index=x_dim, columns=y_dim, values="time_score")
    
    d0 = perf.index.to_numpy(copy=True)
    d2 = perf.columns.to_numpy(copy=True)
    d0, d2 = np.meshgrid(d0, d2)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_zlim([0.0, 1.0])
    ax.set_title("GEMM efficiency")
    
    surf = ax.plot_surface(d0, d2, perf.values.T, cmap='coolwarm', antialiased=True)

    # d1 = anomalies.index.to_numpy(copy=True)
    # d4 = anomalies.columns.to_numpy(copy=True)
    # d1, d4 = np.meshgrid(d1, d4)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel(x_dim)
    # ax.set_ylabel(y_dim)
    # ax.set_title("Anomalies Intensity")

    # the transposition of the values is needed due to how x and y axes are chosen when
    # pivoting the data.
    # surf = ax.plot_surface(d1, d4, anomalies.values.T, cmap='coolwarm', antialiased=True)
    # surf = ax.contour3D(d1, d4, anomalies.values.T, cmap='coolwarm', antialiased=True)

    plt.show()