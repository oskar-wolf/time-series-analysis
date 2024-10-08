# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Seasonality - Harmonic Regression

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# %% set font size for plots
plt.rcParams.update({'font.size': 18})

# %% load data
df = pd.read_csv('AEP_hourly_Sep2012.csv', sep=',')
df['energy'] = df['energy']
df.index = pd.to_datetime(df.datetime, \
    dayfirst=True, infer_datetime_format = True)

# %% estimate the periods
# (1 day = 24 hours, 1 week = 168 hours)
period01=24
period02=168

# %% extract the number hourly samples
T=len(df['energy'])
time=np.arange(1,T+1)

# %% create regressors for the daily period
omega01=2*np.pi/period01
s1=np.cos(omega01*time)
c1=np.sin(omega01*time)

# %% create regressors for the weekly period
omega02=2*np.pi/period02
s2=np.cos(omega02*time)
c2=np.sin(omega02*time)

# %% stack regressors and add intercept
regressors = np.column_stack((s1,c1,s2,c2))
regressors_one = sm.add_constant(regressors)

# %% create and fit the regression model
model = sm.OLS(df['energy'],regressors_one)
results = model.fit()

# %% plot the data and fitted values
plt.figure(figsize=(16,5), dpi=100)

# original data
plt.plot(df. index,df['energy'], color='k', \
    label='Energy consumption')

# fitted values
plt.plot(df.index, results.fittedvalues, color='red', \
    label='$\omega_{1} = 2\pi/24$ and $\omega_{2} = 2\pi/168$')

# add labels, ticks and legend
plt.gca().set(xlabel='Date-Time', \
    ylabel='Consumption (MW)')
plt.xticks(df.index[np.arange(1,len(df)+1, 100)])
plt.legend()
plt.grid()

# show the plot
plt.show()
