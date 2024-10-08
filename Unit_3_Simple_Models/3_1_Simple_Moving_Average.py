# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Simple Moving Average (SMA)

## SMA example - Precipitation

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% load data & change names
fname = 'average-precipitation-1901-2020_br__germany.csv'
df = pd.read_csv(fname, index_col='Category'). \
    rename(columns={'Annual Mean': 'Precipitation'}). \
    drop(columns={'Smoothed'})

# rename index
df.index = df.index.rename('Year')

# %% two simple moving averages
df['SMA5'] = df.Precipitation.rolling(5, min_periods=1).\
    mean().shift(1)
df['SMA10'] = df.Precipitation.rolling(10, min_periods=1).\
    mean().shift(1)


# %% plot the data
# create the figure
fig, axs = plt.subplots(nrows=2, ncols=1, \
    figsize=(16,12), dpi=100)
plt.subplots_adjust(hspace = 0.3)

# define colors
colors = ['k','red','blue']

# plot the data
df.plot(ax=axs[0], color=colors, linewidth=3)

# set title, axes, legend and grid
axs[0].set_title('Precipitation 1901-2020')
axs[0].set(xlabel='Year', ylabel='Rainfall [mm]')
axs[0].legend(labels=['Precipitations',\
    '5-years SMA', '10-years SMA'])
axs[0].grid()

# show a subsection
df.loc[1960:1985].plot(ax=axs[1], color=colors,\
                       linewidth=3)

# set title, axes, legend and grid
axs[1].set_title('Precipitation 1960-1985')
axs[1].set(xlabel='Year', ylabel='Rainfall [mm]')
axs[1].legend(labels=['Precipitations', \
    '5-years SMA', '10-years SMA'])
axs[1].grid()

## AIC, MSE

# %% define a function to calculate the AIC
# y: original data
# yhat: SMA data
# npar: number of parameters (by default equal to 2)

def getAIC(y, yhat, npar=2): 
    N = len(y)
    error = y-yhat
    likelihood = (N/2)*np.log(2*np.pi) + \
        (N/2)*np.log(error.var()) + (N/2)
    result = 2*npar - 2*likelihood
    return result

# %% calculate the AIC values for both SMAs
aic_5 = getAIC(df.Precipitation, df['SMA5'])
aic_10 = getAIC(df.Precipitation, df['SMA10'])

print('SMA(5) - AIC:', aic_5)
print('SMA(10) - AIC', aic_10)

# console output:
# SMA(5) - AIC: -1414.0229514348625
# SMA(10) - AIC: -1402.6497628190984

## SMA forecasting

# %% define a function for SMA out-of-sample forecasting
# data: original data
# q: SMA order
# h: forecast horizon

def getSMAForecast(data, q, h):
    
    # create a list of zeros for h
    # to be forecasted values plus the last
    # q observed values (size of the window)
    fc = np.zeros(q+h)
    
    # fill the first q forecast values
    # with the last q observed data points
    fc[:q] = data[-q:]
    
    # iterate over the h to be forecasted values
    for i in range(h):
        
        # calculate the current forecast as the
        # mean of the q past values
        fc[q+i] = fc[i:i+q].mean()
    
    # remove the observed data from the forecast
    fc = fc[q:]

    return fc

# %% forecast 20 values for based on a SMA(5) model
fc = getSMAForecast(data=df.Precipitation, q=5, h=20)

# %% convert forecast to DataFrame
fc = pd.DataFrame(fc, columns=['MA1'])

# %% append data set with forecasted values
df = pd.concat([df, fc], ignore_index=True)

# %% reset index
df = df.set_index([pd.Index(df.index.values+1901)])

# %% plot the precipitation measurements & SMA(5) forecast
# create figure
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(16,5), dpi=100)

# observations
plt.plot(df.index, df.Precipitation, color='k')

# forecast
plt.plot(df.index, df['SMA5'], color='red')

# set labels, legend and grid
plt.gca().set(xlabel='Year', ylabel='Rainfall [mm]')
plt.legend(labels=['Precipitations','5-year SMA'])
plt.grid()
plt.show()
