# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# ACF and PACF example on GDP time series

# %% import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% load GDP time series data
country = "Saudi Arabia"
df_gdp = pd.read_csv('gapminder_gdp.csv', sep=";", \
    index_col="country")
df_gdp=df_gdp.iloc[:,150:]
gdp_country = df_gdp.loc[country]
gdp_country=gdp_country.dropna()

# %% Differencing data
gdp_country_diff = gdp_country.diff().dropna()
gdp_country_diff.index=gdp_country_diff.index.astype(int)
gdp_country.index=gdp_country.index.astype(int)

# %% Plot ACF & PACF
fig, axes = plt.subplots(2, 1, \
    figsize=(16,10), dpi=100)

# ACF
plot_acf(gdp_country_diff, alpha=0.05, ax=axes[0], \
    title='Autocorrelation function (ACF)')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')

# PACF
plot_pacf(gdp_country_diff, alpha=0.05, ax=axes[1], \
    title='Partial Autocorrelation function (PACF)')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')

# %%
