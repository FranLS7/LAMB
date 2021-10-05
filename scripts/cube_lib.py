import ctypes
import numpy.ctypeslib as ctl
import platform
import logging
import sys
import numpy as np
import time
import os


def initialise_library(nthreads=1):
    # if nthreads == 1:
    #     lib = 'GC_lib'
    # else:
    lib = 'GC_lib_mt'
    try:
        if platform.system() == 'Darwin':
            library = ctypes.CDLL(lib + '.dylib')
        elif platform.system() == 'Windows':
            library = ctypes.CDLL(lib + '.dll')
        elif platform.system() == 'Linux':
            os.environ["KMP_AFFINITY"] = "verbose,granularity=fine,proclist=[0-9],explicit"
            dir_path = os.path.dirname(os.path.realpath(__file__)) + '/../cube/lib'
            print(dir_path)
            library = ctl.load_library(lib + '.so', dir_path + '/' + lib + '.so')
            # library = ctypes.CDLL(lib + '.so')#,'/home/bsc18/bsc18266/Correlation_in_CUDA/correlation_multicuda.so')
            # library = ctypes.CDLL('correlation_c.so')
    except:
        logging.error("Library has not been properly loaded")
        sys.exit(1)
    return library

def compute_cube(min_size, max_size, npoints, nsamples, nthreads, output_file):
    library = initialise_library(nthreads)

    min_size_ = np.ascontiguousarray (min_size, dtype=np.int32)
    max_size_ = np.ascontiguousarray (max_size, dtype=np.int32)
    npoints_ = np.ascontiguousarray (npoints, dtype=np.int32)
    ofile = output_file.encode('utf-8')


    c_int_p = ctypes.POINTER(ctypes.c_int)
    # c_double_p = ctypes.POINTER(ctypes.c_double)
    # c_float_p = ctypes.POINTER(ctypes.c_float)

    print(">> Python: About to enter the C++ function")

    # if nthreads == 1:
    #     library.CubeGEMM.restype = None
    #     library.CubeGEMM.argtypes = [c_int_p, c_int_p, c_int_p, ctypes.c_int,
    #                               ctypes.c_int, ctypes.c_char_p]
    #     print('CALLING CUBEGEMM')
    #     library.CubeGEMM (min_size_.ctypes.data_as(c_int_p),
    #                   max_size_.ctypes.data_as(c_int_p),
    #                   npoints_.ctypes.data_as(c_int_p),
    #                   ctypes.c_int(nsamples),
    #                   ctypes.c_int(nthreads),
    #                   ofile)
    #
    #
    # else:
    library.CubeGEMM_mt.restype = None
    library.CubeGEMM_mt.argtypes = [c_int_p, c_int_p, c_int_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_char_p]
    # print('CALLING CUBEGEMM_MT')
    library.CubeGEMM_mt (min_size_.ctypes.data_as(c_int_p),
                          max_size_.ctypes.data_as(c_int_p),
                          npoints_.ctypes.data_as(c_int_p),
                          ctypes.c_int(nsamples),
                          ctypes.c_int(nthreads),
                          ofile)






    print("")
