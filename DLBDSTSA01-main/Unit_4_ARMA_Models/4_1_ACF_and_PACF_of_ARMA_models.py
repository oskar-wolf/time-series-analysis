# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# ACF and PACF of ARMA models

# %% import moduls
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

## AR(1)

# %% set AR parameters
# for zero-lag, add 1
# for lag 1, e.g. AR(1) model, set phi to 0.8
# (given as negative parameter, see polynomial formula)
ar = np.array([1, -0.8])

# %% create the AR(1) model with specified parameters
arma_model = ArmaProcess(ar=ar, ma=None)

# %% generate data based on the AR(1) model
simulated_ar = arma_model.generate_sample(nsample=10000)

# %% figure preparation
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2, 1, figsize=(16,10), dpi=100)
plt.subplots_adjust(hspace=0.3)

# plot ACF
plot_acf(simulated_ar, alpha=0.05, ax=axes[0])
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')
axes[0].set_title(r'ACF AR(1) Model with $\phi=0.8$')

# plot PACF
plot_pacf(simulated_ar, alpha=0.05, ax=axes[1])
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')
axes[1].set_title(r'PACF AR(1) Model with $\phi=0.8$')

## MA(1)

# %% simulate MA(1) data
ma = np.array([1,-0.8])
arma_model = ArmaProcess(ar=None, ma=ma)
simulated_ar = \
    arma_model.generate_sample(nsample=10000)

# %% figure preparation
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2, 1, figsize=(16,10), dpi=100)
plt.subplots_adjust(hspace=0.3)

# plot ACF
plot_acf(simulated_ar, alpha=0.05, ax=axes[0])
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')
axes[0].set_title(r'ACF AR(1) Model with $\phi=0.8$')

# plot PACF
plot_pacf(simulated_ar, alpha=0.05, ax=axes[1])
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')
axes[1].set_title(r'PACF AR(1) Model with $\phi=0.8$')

## ARMA(1,1)

# %% simulate ARMA(1,1) data
ar = np.array([1,-0.4])
ma = np.array([1,0.8])
arma_model = ArmaProcess(ar=ar, ma=ma)
simulated_ar = \
    arma_model.generate_sample(nsample=10000)

# %% figure preparation
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2, 1, figsize=(16,10), dpi=100)
plt.subplots_adjust(hspace=0.3)

# plot ACF
plot_acf(simulated_ar, alpha=0.05, ax=axes[0])
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')
axes[0].set_title(r'ACF AR(1) Model with $\phi=0.8$')

# plot PACF
plot_pacf(simulated_ar, alpha=0.05, ax=axes[1])
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')
axes[1].set_title(r'PACF AR(1) Model with $\phi=0.8$')


# %%