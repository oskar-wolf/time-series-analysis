# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Autoregressive Integrated Moving Average Models (ARIMA)

# %% import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from gdp_data_preparation import prepare_gdp_data

# %% load data
df_gdp = pd.read_csv('total_gdp_ppp_inflation_adjusted.csv', \
    index_col = 'country')

# %% data preprocessing
gdp_country = prepare_gdp_data(df_gdp, "Spain")

# %% calculate Log time series
df_gdp_raw = pd.to_numeric(gdp_country).apply(np.exp)

# %% plot the time series
plt.figure(figsize=(16,5), dpi=100)

plt.plot(df_gdp_raw)
plt.gca().set(xlabel='Time (years)', \
    ylabel = 'GDP PPP Inflation adjusted (USD)')
plt.xticks(np.arange(min(gdp_country.index), \
    max(gdp_country.index)+1, 10.0))
plt.grid()
plt.show()

# %% ADF test on log GDP Spain
resultADF = adfuller(gdp_country, \
    regression="nc", regresults=True)
print('ADF Statistic: %f' % resultADF[0])
print('p-value: %f' % resultADF[1])

# %% difference the data and reset index
gdp_country_diff = gdp_country.diff().dropna()

# %% ADF test on differenced data
resultADF=adfuller(gdp_country_diff,regression="nc",regresults=True)
print('ADF Statistic: %f' % resultADF[0])
print('p-value: %f' % resultADF[1])

# %% import additional method
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% Plot ACF & PACF differenced data
fig, axes = plt.subplots(2, 1, figsize=(16,10), \
    dpi= 100)
plt.subplots_adjust(hspace = 0.4)

# ACF
plot_acf(gdp_country_diff, alpha=0.05, ax=axes[0], \
    title='ACF of differenced Log GDP of Spain')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')

# PACF
plot_pacf(gdp_country_diff, alpha=0.05, ax=axes[1], \
    title='PACF of differenced Log GDP of Spain')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')

# %% import additional method
from statsmodels.tsa.arima.model import ARIMA

# %% prepare the data
gdp_country_np = pd.to_numeric(gdp_country).values

# %% build and fit an ARIMA(p,d,q) model
order = (1, 1, 0)
model = ARIMA(endog=gdp_country_np, exog=None, \
    trend='t', order=order)
m_ar1 = model.fit()

# %% print the model summary
print(m_ar1.summary())
