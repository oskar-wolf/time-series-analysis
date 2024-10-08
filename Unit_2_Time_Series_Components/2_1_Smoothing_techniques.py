# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Smoothing techniques

## Moving average

# %% import modules
import pandas as pd

# %% read example data
df = pd.read_csv('Global_Temperature.csv', \
    sep=';', index_col='Year')

# %% perform rolling window averaging
# window = window length, centered = type of averaging
roll_temp_mean = df.rolling(window=5, center=True).mean()

## Kernel smoothing

# %% import modules
from statsmodels.nonparametric.kernel_regression import KernelReg
from matplotlib import pyplot as plt

# %% conduct kernel smooting
# base the smoothing on the data index
# set the variable type to 'continuous'
# use a bandwidth of 5
kr = KernelReg(df['Global_Temp'].values, \
    df.index, var_type='c', bw=[5])

# %% extract fitted values
x_pred, x_std = kr.fit(df.index)

# %% plot originial and fitted values
# original values
plt.plot(df.index, df.Global_Temp)

# smoothed values
plt.plot(df.index, x_pred,':', \
    color='blue', label='$b = 5$')

# add a legend
plt.legend()
