# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Centered Moving Average (CMA)

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

# %% moving average
df['CMA3'] = df.Precipitation.rolling(3, \
    center=True, min_periods=1).mean()
df['CMA7'] = df.Precipitation.rolling(7, \
    center=True, min_periods=1).mean()

# %% plot the data
# create the figure
plt.rcParams.update({'font.size': 18})
fig, axs = plt.subplots(nrows=2 ,ncols=1, \
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
                      '3-year CMA','7-year CMA'])
axs[0].grid()

# show a subsection
df.loc[1960:1985].plot(ax=axs[1], \
    color=colors, linewidth=3)

# set title, axes, legend and grid
axs[1].set_title('Precipitation 1960-1985')
axs[1].set(xlabel='Year', ylabel='Rainfall [mm]')
axs[1].legend(labels=['Precipitations',\
                      '3-year CMA','7-year CMA'])
axs[1].grid()

# %%
