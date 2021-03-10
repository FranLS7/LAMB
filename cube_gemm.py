import numpy as np
import pandas as pd

data_dir = "./MC3/timings/3D/"
data_file = "GEMM_cube.csv"

CPU_1FREQ = 4.5e9
CPU_6FREQ = 4e9
FLOPS_DP = 16
NUM_CORES = 6

TPP_1CORE = CPU_1FREQ * FLOPS_DP
TPP_NCORE = CPU_6FREQ * FLOPS_DP * NUM_CORES

class Cube():
    
    def __init__ (self, filename=None, min_size=None, max_size=None, jump_size=None, iterations=None):
        self.filename = filename
        self.min_size = min_size
        self.max_size = max_size
        self.jump_size = jump_size
        self.iterations = iterations
        self.npoints = int((self.max_size - self.min_size) / self.jump_size) + 1
        self.points = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
        
        
    def __compute_flops(self):
        print("hello there, I'm inside __compute_flops")
        points_n = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
        points_n = np.reshape (points_n, newshape=(1, len(points_n)))
        
        points_k = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
        points_k = np.reshape (points_k, newshape=(len(points_k), 1))
        
        points_m = np.arange (self.min_size, self.max_size + 1, self.jump_size, dtype=np.float)
        
        flops = points_k * points_n
        flops = flops[..., np.newaxis] * points_m
        return flops
    
    def __scale_index (self, index):
        return int((index - self.min_size) / jump_size)
        
        
    def form_cube (self, delimiter=", ", ndims=3, mode='min'):
        flops = self.__compute_flops()
        
        self.data = pd.read_csv (self.filename, delimiter=delimiter, index_col=list(range(ndims)))
        self.data = self.data.to_numpy (dtype=np.double, copy=True)
        
        if mode == 'mean':
            self.data = self.data.mean (axis=1)
        elif mode == 'min':
            self.data = self.data.min (axis=1)

        self.data = np.reshape(self.data, newshape=(self.npoints, self.npoints, self.npoints))
        self.data = 2 * flops / (self.data * TPP_1CORE)
        
        
    def extract_point (self, indexes, mode='simple', ndims=3):
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
                
    def save_data (self, filename):
        header = str(self.min_size) + '\n' + str(self.max_size) + '\n' + str(self.jump_size)
        data_to_save = np.reshape(self.data, newshape=(self.npoints ** 3, 1))
        np.savetxt (data_dir + filename, X=data_to_save, fmt='%5.15f', delimiter=',', header=header)
                    
                    
        
        
max_size = 500
jump_size = 20
iterations = 10
        
cubo = Cube (data_dir + data_file, jump_size, max_size, jump_size, iterations)
cubo.form_cube()
xxx = cubo.extract_point([490, 490, 490])
# cubo.save_data('GEMM_eff.csv')






