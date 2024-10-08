# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Seasonality - Harmonics and Superposition

# %% import modules
import numpy as np
import matplotlib.pyplot as plt

# %% set font size for graphs
plt.rcParams.update({'font.size': 18})

# %% create time data
t = np.arange(100)

# %% create an harmonic time series 1
A1 = 1
T1 = 100
x1 = A1*np.sin(2*np.pi*t/T1)

# %% create an harmonic time series 2
A2 = 0.3
T2 = 20
x2 = A2*np.sin(2*np.pi*t/T2)

# %% create an harmonic time series 3
A3 = 0.08
T3 = 5
x3 = A3*np.sin(2*np.pi*t/T3)

# %% combine the harmonic series to a superposition
x = x1 + x2 + x3

# %% visualize the series
fig, (ax1, ax2) = plt.subplots(2, 1, \
    figsize=(16,12), dpi=100)

# plot the three harmonic series
ax1.plot(t, x1)
ax1.plot(t, x2, 'tab:orange')
ax1.plot(t, x3, 'tab:green')
ax1.set_title('Three waves')
ax1.grid()

# plot the superposition series
ax2.plot(t, x, 'tab:red')
ax2.set_title('Superposition')
ax2.grid()
