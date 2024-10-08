# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Time Series Forecasting as a Supervised Learning Problem

## Classification with Random Forest

# %% import modules
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import cesium.featurize as ft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %% load stock sample data
stock_df = yf.download( \
    ['AMZN'], \
    start='2015-01-01', \
    end='2020-12-31', \
    progress=False)
stock_df = stock_df.dropna()
ts = stock_df['Adj Close']

# %% plot stock price time series
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 5), dpi=100)
plt.plot(stock_df['Adj Close'], color='k')
plt.gca().set(xlabel='Date-Time', \
    ylabel='Price (US Dollar)')
plt.grid()
plt.show()

## Feature Extraction

# %% set window size
ws = 5

# %% label data points with binary classes
classes = np.where(ts > ts.shift(), \
    'bull', 'bear')[0:len(ts)-ws]

# %% extract values per sliding window
data = np.lib.stride_tricks.\
    sliding_window_view(ts, 5, writeable=True)\
    [0:len(ts)-ws]

# %% convert array of lists to list of arrays
data = list(map(np.asarray, data))

# %% specify data structure for cesium to work
times = np.tile(np.arange(0, ws), \
    (len(data), 1))

# %% define features to be generated
features_to_extract = [ \
    "amplitude", "percent_beyond_1_std", \
    "percent_close_to_median", "skew", \
    "max_slope","mean", "std", "minimum", \
    "median", "maximum", "median", \
    "weighted_average", "median_absolute_deviation"]

# %% extract features from time series
fset_cesium = ft.featurize_time_series( \
    times = times, \
    values = data, \
    features_to_use = features_to_extract, \
    scheduler = None)

## Classification

# %% split into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split( \
    fset_cesium.values, classes, random_state=21)

# %% build and train the model
rf_clf = RandomForestClassifier( \
    n_estimators=100, \
    max_depth=None, random_state=21)
rf_clf.fit(X_train, y_train)

# %% model evaluation
acc = rf_clf.score(X_test, y_test)
print('Mean Accuracy:', round(acc, 4))

# console output:
# Mean Accuracy: 0.5013

## Regression with Multilayer Perceptron

# %% import module
from sklearn.neural_network import MLPClassifier, MLPRegressor

# %% split the data
X_train, X_test, y_train, y_test = train_test_split( \
    data, ts[ws:], random_state=21)

# %% build and train the model
rgf = MLPRegressor(random_state=21, max_iter=500).\
    fit(X_train, y_train)

# %% model evaluation
acc = rgf.score(X_test, y_test)
print('R²:', round(acc, 4))

# console output:
# R²: 0.998

## MLP classification

# %% split into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split( \
    fset_cesium.values, classes, random_state=21)

# %% build and train the model
clf = MLPClassifier(random_state=21, \
    max_iter=300, activation='tanh', \
    verbose=True).fit(X_train, y_train)

# %% model evaluation
acc = clf.score(X_test, y_test)
print('Mean Accuracy:', round(acc, 4))

# console output:
# Mean Accuracy: 0.5119

# %%