# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Cross Validation (TES)

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# %% load data
df = pd.read_csv('Auto_sales.csv', sep=';', \
    decimal=',', parse_dates=['Date'], \
    index_col='Date')
df = df.dropna()

# %% prepare data splitter
n_splits = 200
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)

# %% prepare MSE objects
mse_ses = np.zeros(n_splits)
mse_des = np.zeros(n_splits)
mse_tes = np.zeros(n_splits)

# %% define MSE function
def calc_mse(model, data, test_idx):

    # extract test data
    y = data[test_idx[0]]

    # make model predictions
    y_hat = model.predict(test_idx[0])

    # calculate MSE
    mse = (y_hat-y)**2

    return mse

# %% iterate over each data split
k=0
for train_index, test_index in tscv.split(df.Auto_sales):
    
    # extract training and testing data
    train_data = df.Auto_sales[train_index]
    test_data = df.Auto_sales[test_index]
    
    # build and fit SES, report MSE
    ses = ES(train_data).fit()
    mse_ses[k] = calc_mse(ses, df.Auto_sales, test_index)

    # build and fit DES, report MSE
    des = ES(train_data, trend='add').fit()
    mse_des[k] = calc_mse(des, df.Auto_sales, test_index)
    
    # build and fit TES, report MSE
    tes = ES(train_data, trend='add', \
        seasonal='mul', seasonal_periods=12).fit()
    mse_tes[k] = calc_mse(tes, df.Auto_sales, test_index)
    
    k+=1

# %% print the results to console
print("SES MSE:", np.sqrt(mse_ses.mean()), '\n',\
      "DES MSE:", np.sqrt(mse_des.mean()), '\n',\
      "TES MSE:", np.sqrt(mse_tes.mean()))

# console output:
# SES MSE: 76.29325185630603 
# DES MSE: 80.55644199404946 
# TES MSE: 49.55971895560938

# %%
