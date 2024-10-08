# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Seasonality - Periodogram

# %% import modules
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# %% set font size for graphics
plt.rcParams.update({'font.size': 18}) 

# %% load data
df = pd.read_csv('AEP_hourly_Sep2012.csv', sep=',')
df['energy'] = df['energy']
df.index = pd.to_datetime(df.datetime, \
    dayfirst=True, infer_datetime_format = True)

# %% generate periodogram
freq, Pxx_spec = signal.periodogram( \
    df['energy'].values, scaling='spectrum')

# %% create graphic
plt.figure(figsize=(16,5), dpi=100)
plt.plot(freq[1:], Pxx_spec[1:]) #f[0]~0, dropped
plt.xlabel('freq (Hz)')
plt.ylabel('PSD')
plt.yscale('log')
plt.grid()

# %% define a function to find the maximium PSDs
# within bounds (f1, f2)
def getMaxPeriodogram(freq, Pxx_spec, f1, f2):
    
    # find the frequencies within the given bounds
    mask = (freq>f1) & (freq<f2)
    
    # extract the maximum PSDs within the bounds
    maxPsd = max(Pxx_spec[mask])
    
    # extract the the frequencies with mamixum PSDs
    freqMax = freq[Pxx_spec==maxPsd]
    
    # convert to periods
    periodMax = 1/freqMax
    
    return({'Maximum PSD':maxPsd, \
            'Frequency': freqMax[0], \
            'Period': periodMax[0]}) 

# %% print the results for given bounds
print(getMaxPeriodogram(freq ,Pxx_spec, 0.002, 0.02))

# console output:
# {'Maximum PSD': 477550.8444001595, \
# 'Frequency': 0.005747126436781609, \
# 'Period': 174.0}

# %% print the results for given bounds
print(getMaxPeriodogram(freq, Pxx_spec, 0.001, 0.06))

# console output:
# {'Maximum PSD': 2246538.2768236673, \
# 'Frequency': 0.041666666666666664, \
# 'Period': 24.0}