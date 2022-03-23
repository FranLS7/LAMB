import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

import time
import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1


class PageSlider(Slider):

    def __init__(self, ax, label, valmin, valmax, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True,
                 dragging=True, orientation='horizontal', **kwargs):

        self.facecolor = kwargs.get('facecolor', "w")
        self.activecolor = kwargs.pop('activecolor', "b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = valmax - valmin

        super(PageSlider, self).__init__(ax, label, valmin, valmax,
                                         valinit=valinit, valfmt=valfmt,
                                         orientation=orientation, **kwargs)

        self.poly.set_visible(False)
        if orientation == 'vertical':
            self.poly = ax.axhspan(valmin, valinit, 0, 1, **kwargs)
            self.hline.set_visible(False)
        else:
            self.poly = ax.axvspan(valmin, valinit, 0, 1, **kwargs)
            self.vline.set_visible(False)

        self.pageRects = []
        for i in range(self.numpages):
            facecolor = self.activecolor if i == valinit else self.facecolor
            r = matplotlib.patches.Rectangle((float(i)/self.numpages, 0), 1./self.numpages, 1,
                                             transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i)/self.numpages+0.5/self.numpages, 0.5, str(i+1),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label='<',
                                                     color=self.facecolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label='>',
                                                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >= self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i + 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i - 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)


if __name__ == "__main__":

    # TRY LOADING ALL THE DATA ON MEMORY AT ONCE
    ndim = 5
    npar = 6
    thresholdAnomalies = 0.05

    data_dir = "multi/timings/MCX/region_2D_combination/"
    base_name = "region"

    data = pd.read_csv(data_dir + "region_1_4.csv")

    # -------------------- FLOPS FIGURE -------------------- #
    flops = pd.pivot(data, index="d1", columns="d4", values="winner_flops")
    fig_flops, ax_flop = plt.subplots()

    plt.title("Min #FLOPs")
    im_flop = ax_flop.imshow(flops, cmap='icefire', vmin=0, vmax=npar-1,
                             extent=[flops.columns.min(),
                                     flops.columns.max(),
                                     flops.index.max(),
                                     flops.index.min()])
    fig_flops.colorbar(im_flop, ax=ax_flop)

    pointsMajorAxis = 21
    ax_flop.set_yticks(np.linspace(flops.index.min(),
                                   flops.index.max(), num=pointsMajorAxis, dtype=int))
    ax_flop.set_xticks(np.linspace(flops.columns.min(),
                                   flops.columns.max(), num=pointsMajorAxis, dtype=int))

    ax_flop.set_xticks(flops.columns, minor=True)
    ax_flop.set_yticks(flops.index, minor=True)

    ax_flop.grid(which='both', color='w', linestyle='-', linewidth=0.15)

    ax1_slider = fig_flops.add_axes([0.15, 0.04, 0.8, 0.04])
    # ax1_slider = fig_flops.add_axes([0.9, 0.1, 0.1, 0.7])
    ax2_slider = fig_flops.add_axes([0.15, 0.00, 0.8, 0.04])
    slider_flop1 = PageSlider(ax1_slider, '1st Dim', valmin=0, valmax=npar,
                              valinit=1, activecolor='orange')  # , orientation='vertical')
    slider_flop2 = PageSlider(ax2_slider, '2nd Dim', valmin=0, valmax=npar-1,
                              valinit=4, activecolor='orange')

    # -------------------- TIME FIGURE -------------------- #
    times = pd.pivot(data, index="d1", columns="d4", values="winner_time")
    fig_time, ax_time = plt.subplots()

    plt.title("Min Time")
    im_time = ax_time.imshow(times, cmap='icefire', vmin=0, vmax=npar-1,
                             extent=[times.columns.min(),
                                     times.columns.max(),
                                     times.index.max(),
                                     times.index.min()])
    fig_time.colorbar(im_time, ax=ax_time)

    ax_time.set_yticks(np.linspace(times.index.min(),
                                   times.index.max(), num=pointsMajorAxis, dtype=int))
    ax_time.set_xticks(np.linspace(times.columns.min(),
                                   times.columns.max(), num=pointsMajorAxis, dtype=int))

    ax_time.set_xticks(times.columns, minor=True)
    ax_time.set_yticks(times.index, minor=True)

    ax_time.grid(which='both', color='w', linestyle='-', linewidth=0.15)

    ax_slider_time1 = fig_time.add_axes([0.15, 0.04, 0.8, 0.04])
    ax_slider_time2 = fig_time.add_axes([0.15, 0.00, 0.8, 0.04])

    slider_time1 = PageSlider(ax_slider_time1, '1st Dim', valmin=0, valmax=npar,
                              valinit=1, activecolor='orange')
    slider_time2 = PageSlider(ax_slider_time2, '2nd Dim', valmin=0, valmax=npar-1,
                              valinit=4, activecolor='orange')

    # -------------------- ANOMALIES FIGURE -------------------- #
    anomalies = pd.pivot(data, index="d1", columns="d4", values="time_score")
    anomalies[anomalies < thresholdAnomalies] = 0.0
    anomalies[anomalies >= thresholdAnomalies] = 1.0
    fig_anomalies, ax_anomalies = plt.subplots()

    plt.title("Presence of Anomalies")
    im_anomalies = ax_anomalies.imshow(anomalies, cmap='Greys', vmin=0.0, vmax=1.0,
                                       extent=[anomalies.columns.min(),
                                               anomalies.columns.max(),
                                               anomalies.index.max(),
                                               anomalies.index.min()])

    ax_anomalies.set_yticks(np.linspace(anomalies.index.min(),
                                        anomalies.index.max(), num=pointsMajorAxis, dtype=int))
    ax_anomalies.set_xticks(np.linspace(anomalies.columns.min(),
                                        anomalies.columns.max(), num=pointsMajorAxis, dtype=int))

    ax_anomalies.set_xticks(anomalies.columns, minor=True)
    ax_anomalies.set_yticks(anomalies.index, minor=True)
    ax_anomalies.grid(which='both', color='w', linestyle='-', linewidth=0.15)

    ax3_slider = fig_anomalies.add_axes([0.15, 0.04, 0.8, 0.04])
    ax4_slider = fig_anomalies.add_axes([0.15, 0.00, 0.8, 0.04])
    slider_an1 = PageSlider(ax3_slider, '1st Dim', valmin=0, valmax=npar,
                            valinit=1, activecolor='orange')
    slider_an2 = PageSlider(ax4_slider, '2nd Dim', valmin=0, valmax=npar-1,
                            valinit=4, activecolor='orange')

    axbox = fig_anomalies.add_axes([0.88, 0.7, 0.1, 0.05])
    txtBox = TextBox(axbox, 'Filter', initial='0.05')

    def update_flops(val):
        i = int(slider_flop1.val)
        j = int(slider_flop2.val)

        di = "d" + str(i)
        dj = "d" + str(j)
        data = pd.read_csv(data_dir + base_name + "_" +
                           str(i) + "_" + str(j) + ".csv")

        flops = pd.pivot(data, index=di, columns=dj, values="winner_flops")

        ax_flop.set_xticks(np.linspace(flops.columns.min(),
                                       flops.columns.max(), num=pointsMajorAxis, dtype=int))
        ax_flop.set_yticks(np.linspace(flops.index.min(),
                                       flops.index.max(), num=pointsMajorAxis, dtype=int))

        ax_flop.set_xticks(flops.columns, minor=True)
        ax_flop.set_yticks(flops.index, minor=True)
        ax_flop.grid(which='both', color='w', linestyle='-', linewidth=0.15)

        # ax_flop.imshow(flops, cmap='icefire',
        #                extent=[flops.columns.min(),
        #                        flops.columns.max(),
        #                        flops.index.max(),
        #                        flops.index.min()])
        im_flop.set_data(flops)
        im_flop.set_extent([flops.columns.min(), flops.columns.max(),
                            flops.index.max(), flops.index.min()])
        ax_flop.figure.canvas.draw()

    def update_times(val):
        i = int(slider_time1.val)
        j = int(slider_time2.val)

        di = "d" + str(i)
        dj = "d" + str(j)

        data = pd.read_csv(data_dir + base_name + "_" +
                           str(i) + "_" + str(j) + ".csv")

        times = pd.pivot(data, index=di, columns=dj, values="winner_time")

        ax_time.set_xticks(np.linspace(times.columns.min(),
                                       times.columns.max(), num=pointsMajorAxis, dtype=int))
        ax_time.set_yticks(np.linspace(times.index.min(),
                                       times.index.max(), num=pointsMajorAxis, dtype=int))

        ax_time.set_xticks(times.columns, minor=True)
        ax_time.set_yticks(times.index, minor=True)
        ax_time.grid(which='both', color='w', linestyle='-', linewidth=0.15)

        im_time.set_data(times)
        im_time.set_extent([times.columns.min(), times.columns.max(),
                            times.index.max(), times.index.min()])

    def update_anomalies(val):
        i = int(slider_an1.val)
        j = int(slider_an2.val)

        di = "d" + str(i)
        dj = "d" + str(j)
        data = pd.read_csv(data_dir + base_name + "_" +
                           str(i) + "_" + str(j) + ".csv")

        anomalies = pd.pivot(data, index=di, columns=dj, values="time_score")
        anomalies[anomalies < thresholdAnomalies] = 0.0
        anomalies[anomalies >= thresholdAnomalies] = 1.0

        ax_anomalies.set_xticks(np.linspace(anomalies.columns.min(),
                                            anomalies.columns.max(), num=pointsMajorAxis, dtype=int))
        ax_anomalies.set_yticks(np.linspace(anomalies.index.min(),
                                            anomalies.index.max(), num=pointsMajorAxis, dtype=int))

        ax_anomalies.set_xticks(anomalies.columns, minor=True)
        ax_anomalies.set_yticks(anomalies.index, minor=True)
        ax_anomalies.grid(which='both', color='w',
                          linestyle='-', linewidth=0.15)

        im_anomalies.set_data(anomalies)
        im_anomalies.set_extent([anomalies.columns.min(), anomalies.columns.max(),
                                 anomalies.index.max(), anomalies.index.min()])
        print("Anomalies updated!")

    def submit(text):
        thresholdAnomalies = float(text)
        print(int(slider_an1.val))
        print(int(slider_an2.val))
        i = int(slider_an1.val)
        j = int(slider_an2.val)
        di = "d" + str(i)
        dj = "d" + str(j)
        data = pd.read_csv(data_dir + base_name + "_" +
                           str(i) + "_" + str(j) + ".csv")
        anomalies = pd.pivot(data, index=di, columns=dj, values="time_score")

        anomalies[anomalies < thresholdAnomalies] = 0.0
        anomalies[anomalies >= thresholdAnomalies] = 1.0

        im_anomalies.set_data(anomalies)
        print("Submit completed!")

    slider_flop1.on_changed(update_flops)
    slider_flop2.on_changed(update_flops)
    slider_time1.on_changed(update_times)
    slider_time2.on_changed(update_times)
    slider_an1.on_changed(update_anomalies)
    slider_an2.on_changed(update_anomalies)
    txtBox.on_submit(submit)
    plt.show()

    # times = data[["d0", "d1", "winner_time"]]
    # times = data.pivot(index="d0", columns="d1", values="winner_time")
    # fig_times = plt.figure()
    # sns.heatmap(times, cmap='icefire', vmin=0, vmax=5)
    # plt.title("Min Time")
    # plt.show()

    # for i in range(ndim - 1):
    #     for j in range (i + 1, ndim):

    # data = pd.read_csv("multi/timings/MCX/region_winners/region_10_1000.csv")

    # d1 = np.unique(data["d1"].to_numpy())
    # d4 = np.unique(data["d4"].to_numpy())

    # flops = data[["d1", "d4", "winner_flops"]]
    # flops = data.pivot(index="d1", columns="d4", values="winner_flops")
    # plt.figure()
    # sns.heatmap(flops, cmap='icefire', vmin=0, vmax=5)
    # plt.title("Min #FLOPs")

    # times = data[["d1", "d4", "winner_time"]]
    # times = data.pivot(index="d1", columns="d4", values="winner_time")
    # plt.figure()
    # sns.heatmap(times, cmap='icefire', vmin=0, vmax=5)
    # plt.title("Min Time")
    # plt.show()
