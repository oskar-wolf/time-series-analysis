# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Box-Jenkins Formalism

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Data preparation

# %% load data
df_temp = pd.read_csv('Global_Temperature.csv', \
    sep=';', index_col='Year')

# %% shape the data to long format
temp = df_temp['Global_Temp'].values.reshape(-1,1)

# Regression model

# %% generate time regressors
T = len(temp)
time = np.arange(1,T+1)/100 #Line
time2 = np.column_stack((time, time**2)) # quadratic
regressors = sm.add_constant(time2)

# %% build and fit the quadratic regression model
model_reg = sm.OLS(temp,regressors)
results_reg = model_reg.fit()

# %% define a function for plotting ACF and PACF
def plot_acf_pacf(data):
    
    # prepare the plot
    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(2, 1, \
        figsize=(16,10), dpi= 100)
    plt.subplots_adjust(hspace=0.3)

    # ACF
    plot_acf(data, alpha=0.05, ax=axes[0],\
        title='Autocorrelation function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Correlation')

    # PACF
    plot_pacf(data, alpha=0.05, ax=axes[1],\
        title='Partial Autocorrelation function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Correlation')

# Box-Jenkins step 1: Oder selection

# %% plot ACF & PACF of the residuals of 
# the regression model
plot_acf_pacf(results_reg.resid)

# Box-Jenkins step 2: Parameter estimation

# %% calculate arithmetic mean of the residuals
mean_residual = results_reg.resid.mean()

# %% define the data for ARMA as the difference 
# between the model residuals and the mean of
# these residuals
X = results_reg.resid-mean_residual

# %% build and fit an AR(1) model
order = (1, 0, 0) # (p, d, q) - d will later be discussed
model = ARIMA(endog=X, exog=None, \
    trend='n', order=order, \
    enforce_stationarity=True, \
    enforce_invertibility=True)
m_ar1 = model.fit()
    
# %% build and fit a MA(4) model
order = (0, 0, 4) # (p, d, q) - d will later be discussed
model = ARIMA(endog=X, exog=None, \
    trend='n', order=order, \
    enforce_stationarity=True, \
    enforce_invertibility=True)
m_ma4 = model.fit()

# %% build and fit an ARMA(1,4) model
order = (1, 0, 4) # (p, d, q) - d will later be discussed
model = ARIMA(endog=X, exog=None, \
    trend='n', order=order, \
    enforce_stationarity=True, \
    enforce_invertibility=True)
m_arma14 = model.fit()

# Box-Jenkins step 3: Model diagnostics

# %% print model summary for AR(1)
print(m_ar1.summary())

# %% print model summary for MA(4)
print(m_ma4.summary())

# %% print model summary for ARMA(1,4)
print(m_arma14.summary())

# %% plot ACF & PACF residuals of the AR(1) model
plot_acf_pacf(m_ar1.resid)

# %% plot ACF & PACF residuals of the MA(4) model
plot_acf_pacf(m_ma4.resid)

# %% plot ACF & PACF residuals of the ARMA(1,4) model
plot_acf_pacf(m_arma14.resid)

# Box-Jenkins step 4: Forecasting

# %% define time steps to be forecasted into the future
steps=20

# %% define exogenous variables for the out of sample forecast
time_oos = np.arange(142, 142+steps)/100
time2_oos = np.column_stack((time_oos, time_oos**2))
regressors_oos = sm.add_constant(time2_oos)

# %% get regression forecast
reg_component = results_reg.predict(exog=regressors_oos)

# %% get AR(1) model forecast
ar_component = m_ar1.forecast(steps=steps)

# %% get temperature forecast
forecasted_temperatures = reg_component + ar_component

# %% add forecast to the original data
oos_index = pd.Int64Index(np.arange(2021, 2021+steps))
all_index = df_temp.index.append(oos_index)
all_forecast = np.append(results_reg.fittedvalues,\
    forecasted_temperatures)

# %% plot data with forecasted values
plt.figure(figsize=(16,5), dpi=100)

# observed data
plt.plot(df_temp.index, df_temp['Global_Temp'], \
    color='k', label='Original data (Delta Temperature)')

# forecasted data
label = '$Temperature_{t}=a+b\cdot '
label += 't+c\cdot t^{2}+AR(1)-Process$'
plt.plot(all_index, all_forecast, '--', color='blue', \
    label=label)

# add artists
plt.gca().set(xlabel='Year', ylabel='Temperature (C°)')
plt.legend()
plt.grid()
plt.show()
