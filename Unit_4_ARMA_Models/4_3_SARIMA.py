# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Seasonal Autoregressive Integrated Moving Average Models (SARIMA)

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX 

# %% load the data and drop missing values
df = pd.read_csv('Auto_sales.csv', sep=';', \
    decimal=',', parse_dates=['Date'], \
    index_col='Date')
df = df.dropna()

# %% plot the raw Data and the ACF
# create the figure
plt.rcParams.update({'font.size':18})
fig, axes = plt.subplots(2, 1, figsize=(16,10), dpi=100)
plt.subplots_adjust(hspace = 0.4)

# add raw data series
axes[0].plot(df.index,df.Auto_sales, \
    color='k', label='Actual vehicle sales')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')
axes[0].grid()

# create ACF plot
plot_acf(df.Auto_sales, alpha=0.05, lags=40, \
    ax=axes[1], title='Autocorrelation function (ACF)')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')

# %% take the first difference
car_diff = df.Auto_sales.diff().dropna()

# %% create ACF and PACF plots for the differenced data
def plot_acf_pacf(data):
    
    # prepare the plot
    fig, axes = plt.subplots(2, 1, \
        figsize=(16,10), dpi=100)
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

plot_acf_pacf(car_diff)
    
# %% SARIMA modeling
X = df.Auto_sales
order = (0, 1, 1) # (p, d, q)
seasonal_order = (1, 0, 0, 12) # (P, D, Q, s)
model=SARIMAX(endog=X, exog=None, \
    trend='n', order=order, \
    seasonal_order=seasonal_order, \
    enforce_stationarity=True, \
    enforce_invertibility=True)
m_sarima6=model.fit()

# %% print model summary
print(m_sarima6.summary())

# %% Plot ACF & PACF SARIMA Residuals
plot_acf_pacf(m_sarima6.resid)