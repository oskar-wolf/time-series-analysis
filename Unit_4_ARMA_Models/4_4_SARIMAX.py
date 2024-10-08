# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Seasonal Autoregressive Integrated Moving Average Models 
# with Exogeneous variables (SARIMAX)

# %% import modules and set seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
np.random.seed(97)

# %% load and subset the data and drop missing values
df = pd.read_csv('Auto_sales.csv', sep=';', \
    decimal=',', parse_dates=['Date'], \
    index_col='Date')
df=df.iloc[120:] # consider data after 2011
df.dropna()

# %% plot raw data & ACF
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2,1,figsize=(16,10), dpi= 100)
plt.subplots_adjust(hspace = 0.4)

# raw data
axes[0].plot(df.index, df.Auto_sales, \
    color='k', label='Actual vehicle sales')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')
axes[0].grid()

# ACF
plot_acf(df.Auto_sales, alpha=0.05, ax=axes[1], \
    title='Autocorrelation function (ACF)', lags=36)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')
axes[1].set_xticks(np.arange(0,37,6))

# %% prepare data for regression model
X = df.Auto_sales
t = np.arange(len(X))
exog_var = np.stack((t, t**2), axis=1)

# %% build and fit the regression model
reg_with_intercept = sm.add_constant(exog_var)
reg_model = sm.OLS(X, reg_with_intercept)
reg_fit = reg_model.fit()

# %% show regression model summary
print(reg_fit.summary())

# %% plot data with the regression curve
plt.figure(figsize=(16,5), dpi=100)

# original data
plt.plot(X, color='k', label='U.S. Automobile Data')

# regression curve
plt.plot(df.index, reg_fit.fittedvalues, \
    color='red', label='Linear Regression')

# add artists
plt.gca().set(xlabel='Year', ylabel='Sales')
plt.legend()
plt.grid()
plt.show()

# %% plot ACF & PACF of regression residuals
def plot_acf_pacf(data):
    
    # prepare the plot
    fig, axes = plt.subplots(2, 1, \
        figsize=(16,10), dpi=100)
    plt.subplots_adjust(hspace=0.3)

    # ACF
    plot_acf(data, alpha=0.05, ax=axes[0], lags=36, \
        title='Autocorrelation function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Correlation')

    # PACF
    plot_pacf(data, alpha=0.05, ax=axes[1], lags=36, \
        title='Partial Autocorrelation function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Correlation')
    axes[1].set_xticks(np.arange(0,37,6))

plot_acf_pacf(reg_fit.resid)

# %% build and fit the SARIMAX model
order = (0, 0, 0)
seasonal_order = (1, 0, 0, 12)
model = SARIMAX(endog=X, exog=exog_var, \
    trend='c', order=order, \
    seasonal_order=seasonal_order, \
    enforce_stationarity=True, \
    enforce_invertibility=True)
m_sarimax = model.fit()

# %% plot ACF & PACF of the SARIMAX residuals
plot_acf_pacf(m_sarimax.resid)

# %% build and fit final SARIMAX model
order = (1, 0, 0)
seasonal_order = (1, 0, 0, 12)
model = SARIMAX(endog=X, exog=exog_var, \
    trend='c', order=order, \
    seasonal_order=seasonal_order, \
    enforce_stationarity=True, \
    enforce_invertibility=True)
m_sarimax = model.fit()

# %% print model summary
print(m_sarimax.summary())

# %% plot ACF & PACF of SARIMAX residuals
plot_acf_pacf(m_sarimax.resid)