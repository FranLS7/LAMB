import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm

# "./MC3/timings/3D/"
data_dir = "./MC3_par/timings/"
data_file = "cube_1500_10th.csv"

CPU_1FREQ = 4.5e9
CPU_6FREQ = 4e9
FLOPS_DP = 16
NUM_CORES = 6

TPP_1CORE = CPU_1FREQ * FLOPS_DP
TPP_NCORE = CPU_6FREQ * FLOPS_DP * NUM_CORES

SERVER_AVX512_FREQ = np.array([2e9, 2e9, 1.8e9, 1.8e9, 1.6e9, 1.6e9, 1.6e9, 1.6e9, 1.5e9, 1.5e9])
SERVER_DP = 16

class Dim():
    def __init__ (self, min_size, max_size, npoints, points):
        self.min_size = min_size
        self.max_size = max_size
        self.npoints = npoints
        self.points = points



class Cube():
    
    # def __init__ (self, filename=None, min_size=None, max_size=None, 
    #               jump_size=None, nthreads=None, iterations=None):
    #     self.filename = filename
    #     self.min_size = min_size
    #     self.max_size = max_size
    #     self.jump_size = jump_size
    #     self.nthreads = nthreads
    #     self.iterations = iterations
    #     self.npoints = int((self.max_size - self.min_size) / self.jump_size) + 1
    #     self.points = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
    #     self.mode = None
    #     self.metric = None
    #     self.unit = None
        
    def __init__ (self, filename):
        self.filename = filename
        self.__readHeader(filename)
        self.data = np.genfromtxt(filename, delimiter=',')[:, 3:]
        self.iterations = self.data.shape[1]
        
        
        
    def __readHeader (self, filename, overhead=2):
        ifile = open (filename, 'r')
        self.nthreads = int(ifile.readline()[overhead:-1])
        
        ranges = np.fromstring(ifile.readline()[overhead:-1], np.int32, sep=',')
        points = np.fromstring(ifile.readline()[overhead:-1], np.float, sep=',')
        points = np.reshape (points, newshape=(len(points), 1))
        self.d0 = Dim(ranges[0], ranges[1], ranges[2], points)
        
        ranges = np.fromstring(ifile.readline()[overhead:-1], np.int32, sep=',')
        points = np.fromstring(ifile.readline()[overhead:-1], np.float, sep=',')
        points = np.reshape (points, newshape=(len(points), 1))
        self.d1 = Dim(ranges[0], ranges[1], ranges[2], points)

        ranges = np.fromstring(ifile.readline()[overhead:-1], np.int32, sep=',')
        points = np.fromstring(ifile.readline()[overhead:-1], np.float, sep=',')
        points = np.reshape (points, newshape=(len(points), 1))
        self.d2 = Dim(ranges[0], ranges[1], ranges[2], points)

        ifile.close()

    def compute_flops(self):        
        flops = self.d1.points * np.transpose(self.d2.points)
        
        d0_points = np.reshape(self.d0.points, newshape=(self.d0.npoints))
        
        flops = flops[..., np.newaxis] * d0_points
        return 2 * flops
        
        
    # def __compute_flops(self):
    #     self.d0.points = np.reshape(self.d0)
        
    #     points_k = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
    #     points_k = np.reshape (points_k, newshape=(len(points_k), 1))
        
    #     points_m = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
        
    #     flops = points_k * points_n
    #     flops = flops[..., np.newaxis] * points_m
    #     return 2 * flops
    
    
    # TODO: FIX THIS FUNCTION. 
    def __scale_index (self, index):
        return int((index - self.min_size) / self.jump_size)
        
    
   
    # def form_cube (self, delimiter=",", ndims=3, mode='min', metric='eff', tpp=TPP_1CORE):
    #     self.data = pd.read_csv (self.filename, delimiter=delimiter, index_col=list(range(ndims)))
    #     self.data = self.data.to_numpy (dtype=np.double, copy=True)
    #     self.mode = mode
    #     self.metric = metric
        
    #     if mode == 'mean':
    #         self.data = self.data.mean (axis=1)
    #     elif mode == 'min':
    #         self.data = self.data.min (axis=1)

    #     self.data = np.reshape(self.data, newshape=(self.npoints, self.npoints, self.npoints))
        
    #     if metric == 'eff' or metric == 'perf':
    #         flops = self.__compute_flops()
    #         self.data = flops / self.data
    #         self.unit = '[FLOPs]'
    #         if metric == 'eff':
    #             self.data /= tpp
    #             self.unit = '[%]'
    
    # Modify the function so that there is a choice between time, eff and perf.
    # mode:
    #   · 'min'  - assign each point its min
    #   · 'mean' - assign each point its mean
    # metric:
    #   · 'time' - assign each point its time
    #   · 'perf' - assign each point its performance
    #   · 'eff'  - assign each point its efficiency
    def form_cube (self, ndims=3, mode='min', metric='eff', tpp_1c=TPP_1CORE):
        self.mode = mode
        self.metric = metric
        
        if mode == 'mean':
            self.data = self.data.mean (axis=1)
        elif mode == 'min':
            self.data = self.data.min (axis=1)
            
        self.data = np.reshape (self.data, newshape=(self.d0.npoints, 
                                                     self.d1.npoints, 
                                                     self.d2.npoints))
        
        if metric == 'eff' or metric == 'perf':
            flops = self.compute_flops()
            self.data = flops / self.data
            self.unit = '[FLOPs]'
            if metric == 'eff':
                self.tpp = (tpp_1c * self.nthreads)
                self.data /= self.tpp
                self.unit = '[%]'
        elif metric == 'time':
            self.unit = '[s]'
        
    # TODO: FIX THIS FUNCTION
    def get_value (self, indexes, mode='simple', ndims=3):
        idx_interpolation = [False] * ndims
        idx = indexes.copy()
        
        if min(indexes) < self.min_size or max(indexes) > self.max_size:
            print ("Error: Indexes out of range")
            return
        else:
            if mode == 'simple':
                for i in range(len(indexes)):
                    idx[i] = self.__scale_index (indexes[1])
                return self.data[tuple(idx)]
                
            elif mode == 'complex':
                for i in range(len(indexes)):
                    if indexes[i] in self.points:
                        idx_interpolation = True
                       
    # TODO: EACH DIMENSION COULD BE DIFFERENT NOW
    def get_diagonal (self):
        return np.array([self.data [i, i, i] for i in range(self.d0.npoints)])
    
    # TODO: THE DIMENSIONS MIGHT NOT MATCH
    def plot_diagonal (self):
        if self.d0.npoints == self.d1.npoints and self.d0.npoints == self.d2.npoints:
            plt.figure()
            ax = sns.lineplot (data=self.get_diagonal());
            
            ax.set (xlabel='m=k=n [' + str(self.d0.min_size) + ':' + str(self.d0.max_size) \
                    + ':' + str(self.d0.npoints) + ']', \
                    ylabel=self.metric + ' ' + self.mode + ' ' + self.unit, \
                    title="Cube's diagonal " + self.metric)
            
            ax.grid()
            plt.show()
        else:
            print('>> Error: There is no diagonal for this object (sizes mismatch)')
        
    
    def get_slice (self, axis, value):
        if axis == 0:
            plane = self.data[value, :, :]
        elif axis == 1:
            plane = self.data[:, value, :]
        elif axis == 2:
            plane = self.data[:, :, value]
        return plane
    
    # TODO: CHECK THE FUNCTION WITH THE NEW AXES
    def plot_slice (self, axis, value):
        plane = self.get_slice (axis, value)            
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        if axis == 0:
            X = self.d1.points
            Y = self.d2.points
            slider_points = self.d0.npoints
        elif axis == 1:
            X = self.d0.points
            Y = self.d2.points
            slider_points = self.d1.npoints
        elif axis == 2:
            X = self.d0.points
            Y = self.d1.points
            slider_points = self.d2.npoints
        
        # X = np.arange (self.min_size, self.max_size + 1, self.jump_size)
        # Y = np.arange (self.min_size, self.max_size + 1, self.jump_size)
        X, Y = np.meshgrid (X, Y)
        
        surf = ax.plot_surface(X, Y, plane, cmap=cm.coolwarm)
        fig.colorbar(surf)
        
        slider_axes = plt.axes([0.3, 0.05, 0.5, 0.03])
        slider = Slider(slider_axes, 'Axis ' + str(axis), 0, slider_points - 1, valinit=value, valfmt='%i')
        
        def update(val):
            plane = self.get_slice (axis, int(slider.val))
            ax.clear()
            surf = ax.plot_surface(X, Y, plane, cmap=cm.coolwarm)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        plt.show()
        
    # TODO: CHECK THE FUNCTION WITH THE NEW AXES
    def save_data (self, filename):
        header = str(self.d0.min_size) + '\n' + str(self.d0.max_size) + '\n' \
        + str(self.d0.npoints) + '\n'\
        + str(self.d1.min_size) + '\n' + str(self.d1.max_size) + '\n' \
        + str(self.d1.npoints) + '\n'\
        + str(self.d2.min_size) + '\n' + str(self.d2.max_size) + '\n' \
        + str(self.d2.npoints) 
        data_to_save = np.reshape(self.data, newshape=(self.d0.npoints *
                                                       self.d1.npoints * 
                                                       self.d2.npoints, 1))
        np.savetxt (filename, X=data_to_save, fmt='%5.15f', delimiter=',', header=header)
                    
                    
        
        
# max_size = 1500
# jump_size = 20
# iterations = 10
        
# cubo = Cube (data_dir + data_file, jump_size, max_size, jump_size, iterations)
# cubo.form_cube(mode='min', metric='perf')
# cubo.plot_slice(axis=0, value=74)


# cubo.plot_diagonal()
# cubo.plot_slice(axis=0, value=24)
# cubo.save_data()



# cubito = Cube (data_dir + "cube1500.csv", jump_size, 1500, jump_size, iterations)
# cubito.form_cube(mode='min', metric='perf')
# cubito.plot_diagonal()
# cubito.plot_slice(axis=0, value=74)

# cubito.save_data(data_dir + 'cube1500_server.csv')

cube = Cube ('../timings/cube_10cores.csv')

cube.form_cube(metric='eff', tpp_1c=(SERVER_AVX512_FREQ[9] * SERVER_DP))
cube.plot_diagonal()
cube.plot_slice(0,21)

# cube.save_data('../timings/10cores_proc.csv')




