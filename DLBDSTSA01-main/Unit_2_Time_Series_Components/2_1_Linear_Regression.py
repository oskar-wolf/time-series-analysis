# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Linear Regression

## Toy example

# %% import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %% create sample data
z = np.array([2,4,6,8,10])
x = np.array([-0.22, -0.03, 0.25, 0.54, 0.78])

# reshape data as column vector
z = z.reshape(-1,1)
x = x.reshape(-1,1)

# %% create linear regression object, fit and prediction
reg = LinearRegression()
reg.fit(z, x)
x_predicted = reg.predict(z)

# %% display parameters
print(reg.intercept_)
print(reg.coef_)

# console output:
# [-0.507]
# [[0.1285]]

# %% plot actual and predicted data
plt.figure(figsize=(16,5), dpi=100)
plt.scatter(z,x,color='0')
plt.plot(z, x_predicted, color='red')
plt.gca().set(xlabel='Z', ylabel='X')
plt.grid()
plt.show()

## Real-data example

# %% import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# %% read the data
df_temp = pd.read_csv('Global_Temperature.csv',sep=';', \
    index_col = 'Year')

# %% shape data in the right format (many rows, one column)
temp = df_temp['Global_Temp'].values.reshape(-1,1)

# %% generate time regressors
T = len(temp)
time = np.arange(1, T+1)/100 #Line
time2 = np.column_stack((time,time**2)) #Quadratic

# %% Model: Regression Temperature vs. Time (Y,X)
# Degrees one and two
# sm does not include the model constant (intercept)
# by default
time_ones = sm.add_constant(time) 
time2_ones = sm.add_constant(time2)

model_01a = sm.OLS(temp,time_ones) #Line
model_01b = sm.OLS(temp,time2_ones) #Quadratic
results_01a = model_01a.fit() #Ordinary Least Squares fit
results_01b = model_01b.fit()

# %% display the summary on the screen
print(results_01a.summary())

# %% display the summary on the screen
print(results_01b.summary())

# %% plot data (with interpolating line)
plt.figure(figsize=(16,5), dpi=100)

# original data
plt.plot(df_temp.index, df_temp['Global_Temp'], color='k',\
    label='Original data (Delta Temperature)')

# fitted values 1
plt.plot(df_temp.index, results_01a.fittedvalues, \
    color='red',label='$X_{t}=a+b\cdot t$')

# fitted values 2
plt.plot(df_temp.index,results_01b.fittedvalues, \
    '--', color='blue',\
        label='$X_{t}=a+b\cdot t+c\cdot t^{2}$')

# set title and annotations
fig_title = 'Global surface temperature relative \
    to 1951-1980 average temperatures'
plt.gca().set(title=fig_title, xlabel='Year', \
    ylabel='Delta (C°)')

# add legend, grid and show the plot
plt.legend()
plt.grid()
plt.show()
