import numpy as np
import pandas as pd
import time

data_dir = '../timings/'
data_file0 = 'parenth_1.5k_5_20_0.csv'
data_file1 = 'parenth_1.5k_5_20_1.csv'

threshold = 0.7
ndim = 4



# --------------------------- Read data from files --------------------------
parenth_0 = np.genfromtxt (data_dir + data_file0, delimiter=',', skip_header=1)
parenth_1 = np.genfromtxt (data_dir + data_file1, delimiter=',', skip_header=1)

iterations = parenth_0.shape[1] - ndim
threshold = int((2 * threshold * iterations - iterations)) 

# ---------------------- Compute flops for each parenth ---------------------
flops_0 = parenth_0[:, 0] * parenth_0[:, 2] * (parenth_0[:, 1] + parenth_0[:, 3])
flops_1 = parenth_1[:, 1] * parenth_1[:, 3] * (parenth_1[:, 0] + parenth_1[:, 2])

# The following line places:
#   · -1 : where parenthesisation_0 has fewer flops to perform
#   ·  0 : when both parenthesisations have same number of flops
#   ·  1 : where parenthesisation_1 has fewer flops to perform
flops_diff = np.sign(flops_0 - flops_1)


# ------------------- Random sort each iteration to compare -----------------
# times_0 = np.sort(parenth_0[:, 4:], axis=1)
# times_1 = np.sort(parenth_1[:, 4:], axis=1)
times_0 = parenth_0[:, ndim:]
times_1 = parenth_1[:, ndim:]

start = time.time()
for i in range (times_0.shape[0]):
    np.random.shuffle(times_0[i])
    np.random.shuffle(times_1[i])
diff_time = time.time() - start
print("Data shuffling finished in: " + str(diff_time))


# The following line places:
#   · -1 : where parenthesisation_0 is faster
#   ·  1 : where parenthesisation_1 is faster
times_diff = np.sign(times_0 - times_1)

# Implement logic to obtain only those cases in which #flops does not lead to
# faster solution. => Anomaly_Detection

times_diff = np.sum(times_diff, axis=1)

anomaly = np.multiply (times_diff, flops_diff)
# anomaly_sign = np.sign(anomaly)

anomaly_index = np.where(anomaly <= -threshold)[0] #anomaly_sign

anomaly_sizes = parenth_0[anomaly_index, :ndim]

flops_rel_diff = np.abs(flops_0[anomaly_index] - flops_1[anomaly_index]) /\
    np.maximum(flops_0[anomaly_index], flops_1[anomaly_index])
    
times_rel_diff = np.abs(np.median(times_0[anomaly_index], axis=1) \
                      - np.median(times_1[anomaly_index], axis=1)) / \
    np.maximum(np.median(times_0[anomaly_index], axis=1), 
               np.median(times_1[anomaly_index], axis=1))








