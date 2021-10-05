import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

data_dir = "../multi/timings/MC4_mt/regions/"
filename = "dims_0.csv"

data = pd.read_csv (data_dir + filename)
data["val"] = data["flops_diff"].apply(lambda x: "blue" if x<0 else "red")


# sorted_data = data.sort_values (["d0","d1","d2","d3","d4"],  \
#                                 ascending = (True,True,True,True,True))

# data_np = data.to_numpy ()

# d0 = data_np [0:41:1, :]
# d1 = data_np [41:66:1, :]
# d2 = data_np [66:107:1, :]
# d3 = data_np [107:146:1, :]
# d4 = data_np [146::1, :]

d0 = data[0:41]
plt.figure()
sns.scatterplot(data=d0, x="d0", y="time_score", c=d0["val"])

d1 = data[41:66]
plt.figure()
sns.scatterplot(data=d1, x="d1", y="time_score", c=d1["val"])

d2 = data[66:107]
plt.figure()
sns.scatterplot(data=d2, x="d2", y="time_score", c=d2["val"])

d3 = data[107:146]
plt.figure()
sns.scatterplot(data=d3, x="d3", y="time_score", c=d3["val"])

d4 = data[146:]
plt.figure()
sns.scatterplot(data=d4, x="d4", y="time_score", c=d4["val"])

