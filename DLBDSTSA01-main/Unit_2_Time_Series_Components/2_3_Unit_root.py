# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Unit root

# %% import modules
import matplotlib.pyplot as plt
import numpy as np

# %% set font size for graphs and seed
plt.rcParams.update({'font.size': 18})
np.random.seed(97)

# %% define a function to generate series
def ar_sim(alpha, sigma=1, x0=0, N=100):
    
    # create an array of zeros
    x = np.zeros(N)

    # set the starting value
    x[0] = x0

    # iterate over all values
    for i in range(N-1):
        
        # calculate the current value of the series
        x[i+1] = alpha * x0 + sigma * np.random.randn()
        
        # set the current value as the last for
        # the next iteration
        x0=x[i+1]
    
    return(x)

# %% simulate three processes
x_stat = ar_sim(alpha=0.8)
x_unit_root = ar_sim(alpha=1)
x_non_stat = ar_sim(alpha=1.1)

# %% plot the series
fig, axs = plt.subplots(3, figsize=(16,18), dpi=100)
plt.subplots_adjust(hspace = 0.3)

# Stationary
axs[0].plot(x_stat,color='k')
axs[0].set_title('Sequence: A')
axs[0].set(xlabel='Time')
axs[0].grid()

# Unit root
axs[1].plot(x_unit_root,color='k')
axs[1].set_title('Sequence B')
axs[1].set(xlabel='Time')
axs[1].grid()

# Non-stationary
axs[2].plot(x_non_stat,color='k')
axs[2].set_title('Sequence C')
axs[2].set(xlabel='Time')
axs[2].grid()
