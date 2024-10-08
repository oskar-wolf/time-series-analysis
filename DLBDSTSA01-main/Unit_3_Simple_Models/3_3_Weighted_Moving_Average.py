# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Weighted Moving Average (WMA)

## Arithmetically decreasing WMA

# %% import modules
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# %% load the data
stock = ['PFE']
stock_df = yf.download(stock, \
    start='2020-03-01', \
    end='2020-07-31', 
    progress=False)

# %% define a function for 
# arithmetically decreasing weights
def getArithmWeights(q):
    
    # create range from 1 to q
    arithmetic_range = np.arange(1,q+1)
    
    # divide the range by the
    # sum of all values
    weights = arithmetic_range/arithmetic_range.sum()
    
    return weights
    
# %% apply weighted moving average
weights = getArithmWeights(q=10)
stock_df['WMA'] = stock_df.Close.rolling(10).\
    apply(lambda x: np.sum(weights*x)).\
        shift(1)

# %% plot the data
# create a figure
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(16,5), dpi=100)

# plot observations
plt.plot(stock_df.index, stock_df.Close, color='k')

# plot WMA values
plt.plot(stock_df.index,stock_df['WMA'], color='red')

# set labels, legend and grid
plt.gca().set(xlabel='Date', ylabel='Price (€)')
plt.legend(labels=['Pfizer stock prices','WMA(10)'])
plt.grid()
plt.show()

## Tuning q

# %% define a function to calculate the AIC
# (in this case just the variance of residuals)
def getAIC(y, yhat, npar=2):
    error = y-yhat
    error = error[~np.isnan(error)]
    res_var = error.var()
    return res_var


# %% maximum order to be evaluated with AIC
Q=25

# %% set starting value for AIC
aic=1e100

# %% find the optimal value for q judging by the AIC
for cur_q in np.arange(1,Q+1):
    
    # set weights as arithmetic progression
    weights = getArithmWeights(cur_q)
    
    # calculate the average of k previous values and
    # multiply by weights
    stock_df['WMA'] = stock_df.Close.rolling(cur_q). \
    apply(lambda x: np.sum(weights*x)). \
        shift(1)
    
    # calculate the AIC
    cur_aic = getAIC(stock_df.Close, stock_df['WMA'])
    
    # if the current AIC is smaller than previous ones...
    if cur_aic<=aic:
        
        # consider the current AIC value the best one
        aic = cur_aic
        
        # assume the current value for q to be 
        # the optimal one
        q = cur_q

# %% report the best value for q and the AIC
print('Best value for q:', q)
print('Corresponding AIC:', aic)

# console output:
# Best value for q: 2
# Corresponding AIC: 0.82552768533932

## Weights by Regression

# %% import modules
from scipy import optimize as opt
from scipy.optimize import Bounds

# %% extract values for y and x according to the equation
# yt = b1xt-1 + b2xt-2 + b3xt-3 + et
Y = np.matrix(stock_df.Close.values[3:]).transpose()
X = np.matrix([ \
    stock_df.Close.values[2:-1], \
    stock_df.Close.values[1:-2], \
    stock_df.Close.values[0:-3]]).\
        transpose()

# %% define the loss function
def norm_squared(beta, Y, X):
    beta=np.matrix(beta).transpose()
    norm_out = (Y-X@beta).transpose()@(Y-X@beta)
    return norm_out[0,0]

# %% wrap the loss function in a lambda function
# to meet the format that is expected by the
# optimization function in scipy
objective = lambda x: norm_squared(x, Y, X)

# %% start with three equal weights
# (repeating 1/3 for 3 times)
start_weights = np.tile(1/3, 3)

# %% set boundaries for the weights to be between 0 and 1
bounds = Bounds(np.tile(0,3).tolist(), np.tile(1,3).tolist())

# %% define the constraint that the sum of weights
# must be 1
eq_cons = {'type':'eq', 'fun':lambda x: sum(x)-1}

# %% minimization of the loss function
# with respect to the weights
res = opt.minimize(objective, start_weights, \
    method='SLSQP', constraints=eq_cons, \
    options={'disp':True}, bounds=bounds)

# %% extract the weights
weights = res.x[::-1]
print(weights)

# console output:
# array([0.06571588, 0.15782975, 0.77645437])

# %% apply weighted moving average
stock_df['WMA'] = stock_df.Close.rolling(3).\
    apply(lambda x: np.sum(weights*x)).\
        shift(1)

# %% plot the data
# create a figure
plt.figure(figsize=(16,5), dpi=100)

# plot observations
plt.plot(stock_df.index,stock_df.Close, color='k')

# plot WMA data
plt.plot(stock_df.index,stock_df['WMA'], color='red')

# set labels, legend and grid
plt.gca().set(xlabel='Date', ylabel='Price (€)')
plt.legend(labels=['Pfizer stock prices','Optimal WMA(3)'])
plt.grid()
plt.show()
