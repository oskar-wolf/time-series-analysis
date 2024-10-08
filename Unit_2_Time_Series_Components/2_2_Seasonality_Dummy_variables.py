# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Seasonality - Dummy variables

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.nonparametric.kernel_regression \
    import KernelReg

# %% set the font size for graphs
plt.rcParams.update({'font.size': 18})

# %% load data
url = "https://data.bts.gov/resource/crem-w557.json"
df = pd.read_json(url)

# %% select columns, rows and drop missing values
df = df[["date", "auto_sales"]].dropna()
df = df[df.date >= pd.to_datetime("2000-01-01")]

# %% divide sales by 1000
df.auto_sales = df.auto_sales / 1000

# %% plot the data
plt.figure(figsize=(16,5), dpi=100)
plt.plot(df.date, df.auto_sales, color='k')
plt.gca().set(title='Total automobile monthly \
        sales in the US (in thousend)', \
    xlabel='Date', ylabel='Number of automobiles (k)')
plt.show()

# %% detrending by kernel smoothing
# extract the number of samples
T=len(df.auto_sales)

# create and fit the kernel regression model
kr = KernelReg(df.auto_sales, np.arange(T), 'c', bw=[7])
y_pred, y_std = kr.fit(np.arange(T))

# %% create a plot
plt.figure(figsize=(16,5), dpi=100)

# original data

plt.plot(df.date, df.auto_sales, \
    color='k', label='Actual vehicle sales')

# fitted data
plt.plot(df.date, y_pred, '--', \
    color='red', label='KS, $b = 7$')

# set describing artists and show the plot
plt.gca().set(xlabel='Date', \
    ylabel='Number of automobiles (k)')
plt.legend()
plt.grid()
plt.show()

# %% remove the trend from the data
detrended_data = df.auto_sales - y_pred

# %% create dummy variables for each month
D = pd.get_dummies(pd.DatetimeIndex(df.date).month)

# %% collect dummies in a numpy array
regressors = D.to_numpy()

# %% create and fit a regression model
model = sm.OLS(detrended_data, regressors)
results = model.fit()

# %% print the regression summary
print(results.summary())

# %% create a plot
plt.figure(figsize=(16,5), dpi=100)

# detrended data
plt.plot(df.date, detrended_data, color='k', \
    label='Detrended actual vehicle sales')

# fitted values
plt.plot(df.date, results.fittedvalues, \
    color='red', label='Regression seasonal \
        dummy variables')

# set artists and show the plot
plt.gca().set(xlabel='Date', \
              ylabel='Number of automobiles (k)')
plt.legend()
plt.grid()
plt.show()
