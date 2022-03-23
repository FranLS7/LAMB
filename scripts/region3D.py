import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    data_dir = "multi/timings/MCX/region_2D_combination/"
    base_name = "region"

    x_dim = "d2"
    y_dim = "d4"

    data = pd.read_csv(data_dir + "region_2_4.csv")
    anomalies = data.pivot(index=x_dim, columns=y_dim, values="time_score")

    d1 = anomalies.index.to_numpy(copy=True)
    d4 = anomalies.columns.to_numpy(copy=True)
    d1, d4 = np.meshgrid(d1, d4)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_title("Anomalies Intensity")

    # the transposition of the values is needed due to how x and y axes are chosen when
    # pivoting the data.
    surf = ax.plot_surface(d1, d4, anomalies.values.T, cmap='coolwarm', antialiased=True)
    # surf = ax.contour3D(d1, d4, anomalies.values.T, cmap='coolwarm', antialiased=False)

    plt.show()