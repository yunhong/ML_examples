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

train_slice = slice(None, "1986-12-25")
valid_slice = slice("1987-01-01", "1988-12-25")
test_slice = slice("1989-01-01", None)

X_train, y_train = X.loc[train_slice], y.loc[train_slice]
X_valid, y_valid = X.loc[valid_slice], y.loc[valid_slice]
X_test, y_test = X.loc[test_slice], y.loc[test_slice]

def multilevel_df_to_ndarray(df):
    shape = [-1] + [len(level) for level in df.columns.remove_unused_levels().levels]
    return df.values.reshape(shape)

X_train_3D = multilevel_df_to_ndarray(X_train)
X_test_3D = multilevel_df_to_ndarray(X_test)
X_valid_3D = multilevel_df_to_ndarray(X_valid)

from sklearn.metrics import mean_absolute_error

def naive(X):
    return X.iloc[:, -1]

y_pred_naive = naive(X_valid)
mean_absolute_error(y_valid, y_pred_naive)

def ema(X, span):
    return X.T.ewm(span=span).mean().T.iloc[:, -1]

y_pred_ema = ema(X_valid, span=10)
mean_absolute_error(y_valid, y_pred_ema)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_linear = lin_reg.predict(X_valid)
mean_absolute_error(y_valid, y_pred_linear)

def plot_predictions(*named_predictions, start=None, end=None, **kwargs):
    day_range = slice(start, end)
    plt.figure(figsize=(10,5))
    for name, y_pred in named_predictions:
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        plt.plot(y_pred[day_range], label=name, **kwargs)
    plt.legend()
    plt.show()
 
plot_predictions(("Target", y_valid),
                 ("Naive", y_pred_naive),
                 ("EMA", y_pred_ema),
                 ("Linear", y_pred_linear),
                 end=365)



