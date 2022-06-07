import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras
import time

print("python", sys.version)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

assert sys.version_info >= (3, 5) # Python ≥3.5 required
assert tf.__version__ >= "2.0"    # TensorFlow ≥2.0 required

dataset_path = keras.utils.get_file(
    "daily-minimum-temperatures-in-me.csv",
    "https://raw.githubusercontent.com/ageron/tf2_course/master/datasets/daily-minimum-temperatures-in-me.csv")
    
temps = pd.read_csv(dataset_path,
                    parse_dates=[0], index_col=0)

temps.info()

temps.head()

temps.plot(figsize=(10,5))
#plt.show()

temps.loc["1984-12-29":"1985-01-02"]

temps = temps.asfreq("1D", method="ffill")
temps.loc["1984-12-29":"1985-01-02"]

def add_lags(series, times):
    cols = []
    column_index = []
    for time in times:
        cols.append(series.shift(-time))
        lag_fmt = "t+{time}" if time > 0 else "t{time}" if time < 0 else "t"
        column_index += [(lag_fmt.format(time=time), col_name)
                        for col_name in series.columns]
    df = pd.concat(cols, axis=1)
    df.columns = pd.MultiIndex.from_tuples(column_index)
    return df

X = add_lags(temps, times=range(-30+1,1)).iloc[30:-5]
y = add_lags(temps, times=[5]).iloc[30:-5]

X.head()
y.head()


