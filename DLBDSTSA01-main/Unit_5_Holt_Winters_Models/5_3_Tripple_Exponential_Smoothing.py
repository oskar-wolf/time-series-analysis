# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Tripple Exponential Smoothing (TES)

# %% import modules
import pandas as pd
import matplotlib
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
tes_auto = ES(df.iloc[:-h, :].Auto_sales, \
    trend='add', seasonal='mul', \
        seasonal_periods=12).fit()

# %% print model summary
print(tes_auto.summary())

# %% add fitted values to data frame
df['tes_fit'] = tes_auto.fittedvalues

# %% plot the results
# create a figure and set parameters
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(16, 5), dpi=100)

# observed data
plt.plot(df.index, df.Auto_sales, \
    color='k', label='Observed data')

# TES model
plt.plot(df.index, df.tes_fit, \
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
df.tes_fit.iloc[-h:] = tes_auto.forecast(steps=h)

# %% plot the forecasts (and observations)
plt.figure(figsize=(16, 5), dpi=100)

# last 20 observations
plt.plot(df.Auto_sales[-20:], \
    color='k', label='Observed')

# TES forecasts
plt.plot(df.tes_fit[-20:], \
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
tes_damp = ES(df.iloc[:-h, :].Auto_sales, \
    trend='add', seasonal='mul', \
    damped_trend=True, \
        seasonal_periods=12).fit(damping_trend=0.7)

# %% add fitted values to data frame
df['tes_damp'] = tes_damp.fittedvalues

# %% forecast the values and save them
# to the data frame
df.tes_damp.iloc[-h:] = tes_damp.forecast(steps=h)

# %% plot the forecasts (and observations)
plt.figure(figsize=(16, 5), dpi=100)

# last 20 observations
plt.plot(df.Auto_sales[-20:], \
    color='k', label='Observed')

# TES forecasts
plt.plot(df.tes_fit[-20:], \
    color='r', label='TES')

# damped TES forecasts
plt.plot(df.tes_damp[-20:], \
    color='b', label='Damped TES, $\phi=0.7$')

# indicate forcasting start point
plt.vlines(df.index[-h], 0, 400, \
    linestyles='dotted', lw=4)

# set artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.grid()
plt.legend()
plt.show()