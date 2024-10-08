# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Double Exponential Smoothing (DES)

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

# %% load data
df = pd.read_csv('Auto_sales.csv', sep=';', \
    decimal=',', parse_dates=['Date'], \
    index_col='Date')
df = df.dropna()

# %% set the forecast horizon
h = 5

# %% build and fit the model
des_auto = ES(df.iloc[:-h, :].Auto_sales, \
    trend='add').fit()

# %% print model summary
print(des_auto.summary())

# %% add fitted values to data frame
df['des_fit'] = des_auto.fittedvalues

# %% plot the results
# create a figure and set parameters
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(16, 5), dpi=100)

# observed data
plt.plot(df.index, df.Auto_sales, \
    color='k', label='Observed data')

# DES model
plt.plot(df.index, df.des_fit, \
    color='r', label='Fitted values')

# set artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.legend()
plt.grid()
plt.show()

## Forecast

# %% forecast the values and save them
# to the data frame
df.des_fit.iloc[-h:] = des_auto.forecast(steps=h)

# %% plot the forecasts (and observations)
plt.figure(figsize=(16, 5), dpi=100)

# last 20 observations
plt.plot(df.Auto_sales[-20:], \
    color='k', label='Observed')

# DES forecasts
plt.plot(df.des_fit[-20:], \
    color='r', label='Forecast')

# indicate forcasting start point
plt.vlines(df.index[-h], 0, 400, \
    linestyles='dotted', lw=4)

# set artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.grid()
plt.legend()
plt.show()

## Damping

# %% build and fit the model
des_damp = ES(df.iloc[:-h, :].Auto_sales, \
    trend='add', damped_trend=True).\
        fit(damping_trend=0.7)

# %% print model summary
print(des_damp.summary())

# %% add fitted values to data frame
df['des_damp_fit'] = des_damp.fittedvalues

# %% forecast the values and save them
# to the data frame
df.des_damp_fit.iloc[-h:] = des_damp.forecast(steps=h)

# %% plot the forecasts (and observations)
plt.figure(figsize=(16, 5), dpi=100)

# last 20 observations
plt.plot(df.Auto_sales[-20:], \
    color='k', label='Observed')

# DES forecasts
plt.plot(df.des_fit[-20:], \
    color='r', label='DES')

# DES damped forecasts
plt.plot(df.des_damp_fit[-20:], \
    color='b', label='Damped DES, $\phi=0.7$')

# indicate forcasting start point
plt.vlines(df.index[-h], 0, 400, \
    linestyles='dotted', lw=4)

# set artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.grid()
plt.legend()
plt.show()
