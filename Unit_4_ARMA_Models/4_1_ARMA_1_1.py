# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Simulating a series using an ARMA(1,1) model

# %% import modules
import matplotlib.pyplot as plt
import numpy as np

# %% define a function to simulate an ARMA(1,1) process
def arma_sim(x0, phi, theta, sigma, N):
    
    # generate error terms
    epsilon = sigma*np.random.randn(N+1)
    
    # create a list to receive the series data
    # and set the starting value of the series
    x = [x0]
    
    # iterate over all time steps
    for t in np.arange(1,N):
        
        # calculate the current value of the series
        # according to the formula for ARMA(1,1)
        xt = phi*x[t-1]+epsilon[t]+theta*epsilon[t-1]
        
        # save the current value to the series list
        x = np.append(x, xt)

    return x

# %% generate the series
np.random.seed(97)
x_arma = arma_sim(x0=0, phi=0.3, theta=0.4, \
    sigma=1, N=100)

# %% plot the series
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(16,5), dpi=100)
plt.plot(x_arma,color='k')
plt.gca().set(xlabel='Time', ylabel='ARMA(1,1) values')
plt.grid()
plt.show()
