# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Simple Exponential Smoothing (SES)

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

# %% load data
df = pd.read_csv('Auto_sales.csv', sep=';', \
    decimal=',', parse_dates=['Date'], \
    index_col='Date')
df = df.dropna()

# %% define initial level, X_hat_1 and 
# smoothing parameter alpha
Xhat1 = df.Auto_sales[0]
alpha1 = 0.9

# %% build and fit the model
ses_0_9 = ES(df.iloc[:-5, :].Auto_sales, \
    initialization_method="known", \
    initial_level=Xhat1).fit(\
        smoothing_level=alpha1, \
        optimized=False)

# %% add fitted values to data frame
df['ses_0_9_fit'] = ses_0_9.fittedvalues

# %% build and fit second model with
# different smoothing paramaeter alpha
alpha2 = 0.2
ses_0_2 = ES(df.iloc[:-5, :].Auto_sales, \
    initialization_method="known", \
    initial_level=Xhat1).fit(\
        smoothing_level=alpha2, \
        optimized=False)                       

# %% add fitted values to data frame
df['ses_0_2_fit'] = ses_0_2.fittedvalues

# %% plot the results
# create a figure and set parameters
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(16, 5), dpi=100)

# observed data
plt.plot(df.index, df.Auto_sales, \
    color='k', label='Automobile data')

# SES model (alpha=0.9)
plt.plot(df.index, df.ses_0_9_fit, \
    color='r', label=r'ES, $\alpha_{1}=0.9$')

# SES model (alpha=0.2)
plt.plot(df.index, df.ses_0_2_fit, \
    color='b', label=r'ES, $\alpha_{2}=0.2$')

# set artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.grid()
plt.legend()
plt.show()

## Forecast

# %% set the forecast horizon
h = 5

# %% forecast the values and save them
# to the data frame
df.ses_0_9_fit.iloc[-h:] = ses_0_9.forecast(steps=h)
df.ses_0_2_fit.iloc[-h:] = ses_0_2.forecast(steps=h)

# %% plot the forecasts (and observations)
plt.figure(figsize=(16, 5), dpi=100)

# last 20 observations
plt.plot(df.Auto_sales[-20:], \
    color='k', label='Automobile data')

# SES forecasts with alpha=0.9
plt.plot(df.ses_0_9_fit[-20:], color='r', \
    label=r'SES Prediction, $\alpha_{1}=0.9$')

# SES forecats with alpha=0.2
plt.plot(df.ses_0_2_fit[-20:], color='b', \
    label=r'SES Prediction, $\alpha_{2}=0.2$')

# indicate forcasting start point
plt.vlines(df.index[-h], 0, 400, \
    linestyles='dotted', lw=4)

# set artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.grid()
plt.legend()
plt.show()

## Finding parameters by OLS

# %% build and fit the model using least squares
ses_ols = ES(df.Auto_sales).fit(method='ls')

# %% print model summary
print(ses_ols.summary())

# %% plot the fitted series
plt.figure(figsize=(16, 5), dpi=100)

# observed data
plt.plot(df.index, df.Auto_sales, color='k')

# fitted data
plt.plot(df.index, ses_ols.fittedvalues, color='r')

# artists
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.grid()
plt.show()

# %%
