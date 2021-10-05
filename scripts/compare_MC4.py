import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import subprocess


def flops_parenth_0 (dims):
    flops = dims[:, 0] * dims[:, 1] * dims[:, 2] + \
            dims[:, 0] * dims[:, 2] * dims[:, 3] + \
            dims[:, 0] * dims[:, 3] * dims[:, 4]
    return 2 * flops


def flops_parenth_1 (dims):
    flops = dims[:, 1] * dims[:, 2] * dims[:, 3] + \
            dims[:, 0] * dims[:, 1] * dims[:, 3] + \
            dims[:, 0] * dims[:, 3] * dims[:, 4]
    return 2 * flops


def flops_parenth_2 (dims):
    flops = dims[:, 1] * dims[:, 2] * dims[:, 3] + \
            dims[:, 1] * dims[:, 3] * dims[:, 4] + \
            dims[:, 0] * dims[:, 1] * dims[:, 4]
    return 2 * flops


def flops_parenth_3 (dims):
    flops = dims[:, 2] * dims[:, 3] * dims[:, 4] + \
            dims[:, 1] * dims[:, 2] * dims[:, 4] + \
            dims[:, 0] * dims[:, 1] * dims[:, 4]
    return 2 * flops


def flops_parenth_4 (dims):
    flops = dims[:, 0] * dims[:, 1] * dims[:, 2] + \
            dims[:, 2] * dims[:, 3] * dims[:, 4] + \
            dims[:, 0] * dims[:, 2] * dims[:, 4]
    return 2 * flops



def compute_flops(dims, parenth):
    if parenth == 0:
        return flops_parenth_0 (dims)
    elif parenth == 1:
        return flops_parenth_1 (dims)
    elif parenth == 2:
        return flops_parenth_2 (dims)
    elif parenth == 3:
        return flops_parenth_3 (dims)
    elif parenth == 4:
        return flops_parenth_4 (dims)
    elif parenth == 5:
        return flops_parenth_4 (dims)
    else:
        print("There is an error with the number of parenthesisations!")


threshold = 0.8
ndim = 5
nparenth = 6
nthreads = 10
iter_val = 20
lo_margin = 0.10

data_dir = '../multi/timings/MC4_mt/'
anomalies_filename = 'anomalies.csv'
validation_filename = 'validated_anomalies.csv'
filename = 'gen_anomalies_'
data_files = [filename + str(i) + ".csv" for i in range(nparenth)]

data = []
flops = []
times = []
for i in range(nparenth):
    data.append(np.genfromtxt(data_dir + data_files[i], delimiter=",", skip_header=1))
    flops.append(compute_flops(data[i], i))
    times.append(data[i][:, ndim:])

iterations = times[0].shape[1]
threshold = int((2 * threshold * iterations - iterations))


# start = time.time()
# for i in range(nparenth):
#     for j in range(times[i].shape[0]):
#         np.random.shuffle(times[i][j])
# print("Data shuffling finished in: " + str(time.time() - start))


anomaly = np.zeros((times[0].shape[0], nparenth, nparenth), dtype=np.float)

flops_rel_diff = []
times_rel_diff = []
total_metrics = ndim + 2 + 2
anomalies = np.zeros([1, total_metrics])
start = time.time()
for i in range(0, nparenth):
    for j in range(i + 1, nparenth):
        flops_diff = np.sign (flops[i] - flops[j])
        times_diff = np.sum(np.sign (times[i] - times[j]), axis=1)
        anomaly[:, i, j] = np.multiply (times_diff, flops_diff)
        anomaly_index = np.where (anomaly[:, i, j] <= -threshold)[0]

        anomalies_aux = np.zeros((len(anomaly_index), total_metrics), dtype=np.float)
        anomalies_aux [:, :ndim] = data[i][anomaly_index, :ndim]
        anomalies_aux [:, ndim] = i
        anomalies_aux [:, ndim + 1] = j

        flops_rel_diff.append(np.abs(flops[i][anomaly_index] - flops[j][anomaly_index]) /\
                              np.maximum(flops[i][anomaly_index], flops[j][anomaly_index]))

        times_rel_diff.append(np.abs(np.median(times[i][anomaly_index], axis=1) \
                                   - np.median(times[j][anomaly_index], axis=1)) /\
                              np.maximum(np.median(times[i][anomaly_index], axis=1), \
                                         np.median(times[j][anomaly_index], axis=1)))

        anomalies_aux [:, ndim + 2] = np.abs(flops[i][anomaly_index] - flops[j][anomaly_index]) /\
                              np.maximum(flops[i][anomaly_index], flops[j][anomaly_index])

        anomalies_aux [:, ndim + 3] = np.abs(np.median(times[i][anomaly_index], axis=1) \
                                   - np.median(times[j][anomaly_index], axis=1)) /\
                              np.maximum(np.median(times[i][anomaly_index], axis=1), \
                                         np.median(times[j][anomaly_index], axis=1))

        anomalies = np.vstack ((anomalies, anomalies_aux))
print("Double loop finished in: " + str(time.time() - start))
anomalies = anomalies [1:, :]
sorted_anomalies = anomalies[np.argsort(-anomalies[:, 8], axis=0)]

# hist = np.histogram(anomalies[:, 8], bins=20)
hist_anomalies = plt.hist (anomalies[:, 8], bins='auto')



anomalies_raw = np.zeros([1, total_metrics + iterations * 2])
start = time.time()
for i in range(0, nparenth):
    for j in range(i + 1, nparenth):
        flops_diff = np.sign (flops[i] - flops[j])
        times_diff = np.sum(np.sign (times[i] - times[j]), axis=1)
        anomaly[:, i, j] = np.multiply (times_diff, flops_diff)
        anomaly_index = np.where (anomaly[:, i, j] <= -threshold)[0]

        anomalies_aux = np.zeros((len(anomaly_index), total_metrics + iterations * 2), dtype=np.float)
        anomalies_aux [:, :ndim] = data[i][anomaly_index, :ndim]
        anomalies_aux [:, ndim] = i
        anomalies_aux [:, ndim + 1] = j

        flops_rel_diff.append(np.abs(flops[i][anomaly_index] - flops[j][anomaly_index]) /\
                              np.maximum(flops[i][anomaly_index], flops[j][anomaly_index]))

        times_rel_diff.append(np.abs(np.median(times[i][anomaly_index], axis=1) \
                                   - np.median(times[j][anomaly_index], axis=1)) /\
                              np.maximum(np.median(times[i][anomaly_index], axis=1), \
                                         np.median(times[j][anomaly_index], axis=1)))

        anomalies_aux [:, ndim + 2] = np.abs(flops[i][anomaly_index] - flops[j][anomaly_index]) /\
                              np.maximum(flops[i][anomaly_index], flops[j][anomaly_index])

        anomalies_aux [:, ndim + 3] = np.abs(np.median(times[i][anomaly_index], axis=1) \
                                   - np.median(times[j][anomaly_index], axis=1)) /\
                              np.maximum(np.median(times[i][anomaly_index], axis=1), \
                                         np.median(times[j][anomaly_index], axis=1))
        anomalies_aux [:, total_metrics:total_metrics + iterations] = times[i][anomaly_index]
        anomalies_aux [:, total_metrics + iterations:] = times[j][anomaly_index]

        anomalies_raw = np.vstack ((anomalies_raw, anomalies_aux))

anomalies_raw  = anomalies_raw [1:, :]
sorted_anomalies_raw = anomalies_raw[np.argsort(-anomalies_raw[:, 8], axis=0)]
# This line sorts the anomalies, in descending order, by the relative diff in exec times


header = "d0, d1, d2, d3, d4, parenth_i, parenth_j, flops_diff, time_score"

np.savetxt (data_dir + anomalies_filename, X=sorted_anomalies, fmt='%5.15f', delimiter=',', header=header)

cmd = './../multi/bin/MC4_val.x ' + str(iter_val) + ' ' + str(nthreads) + ' ' + \
        str(lo_margin) + ' ' + \
        data_dir + anomalies_filename + ' ' + data_dir + validation_filename
subprocess.call(cmd, shell=True)

# stats_flops = []
# stats_times = []

# for i in range(0, nparenth):
#     for j in range(i + 1, nparenth):
#         stats_flops.append()
# anomaly_index = np.where (anomaly <= -threshold)[0]




















