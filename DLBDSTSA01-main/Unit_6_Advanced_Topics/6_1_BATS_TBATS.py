# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# BATS and TBATS

# %% import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbats import TBATS, BATS
import numpy as np
import pickle

# %% load and prepare the sample data
df = pd.read_csv('AEP_hourly_full.csv', sep=',')
df = df[(df.Datetime >= '2011-02-01 00:00:00') & \
    (df.Datetime < '2011-05-21 00:00:00')]
df = df.rename(columns={'AEP_MW': 'energy'})
df.index = pd.to_datetime(df.Datetime, dayfirst=True, \
    infer_datetime_format=True)
df = df.sort_index()

# %% split the data
X_train = df.iloc[:-(20*24)].energy
X_test = df.iloc[-(20*24):].energy

## BATS

# %% build a BATS model
bats_estimator = BATS(
    seasonal_periods=[24, 169],
    use_arma_errors=True,  # consider ARMA
    use_box_cox=True, # consider Box-Cox Trans.
    use_damped_trend=False, # no damping 
    use_trend = False, # no trend
    n_jobs=1 # number of parallel jobs
)

# %% fit the model
bats_model = bats_estimator.fit(X_train)

# %% write object to file
with open('bats_model.pkl', 'wb') as f:
    pickle.dump(bats_model, f)

# %% load object from file
# with open('bats_model.pkl', 'rb') as f:
#     bats_model = pickle.load(f)

# %% print the model summary
print(bats_model.summary())

# %% extract fitted values and add to Data Frame
bats_fit = pd.DataFrame(bats_model.y_hat, \
    columns=['bats_fit'], index=df.index[:len(X_train)])
df['bats_fit'] = bats_fit

# %% use the model to forecast
fc = bats_model.forecast(steps=len(X_test))
fc = pd.DataFrame(fc, columns=['bats_forecast'], \
    index=df.index[len(X_train):])
df['bats_forecast'] = fc

# %% plot the fitted/forecasted data
plt.rcParams.update({'font.size': 14})
fig, (ax1, ax2) = plt.subplots(2, \
    figsize=(16, 10), dpi=100)

# observed data
ax1.plot(df.index, df.energy, \
    color='k', label='Actual')

# fitted values
ax1.plot(df.index, df.bats_fit, \
    color='b', label='Fitted')

# forecasted values
ax1.plot(df.index, df.bats_forecast, \
    color='r', label='Forecast')

# set artists
ax1.set_xlabel('Date')
ax1.set_ylabel('Consumption (MW)')
ax1.grid()
ax1.legend()

# second plot - zoom to 10 days
df_zoom = df[(df.Datetime >= '2011-04-20 00:00:00') & \
    (df.Datetime < '2011-05-10 00:00:00')]

# observed data
ax2.plot(df_zoom.index, df_zoom.energy, \
    color='k', label='Actual')

# fitted values
ax2.plot(df_zoom.index, df_zoom.bats_fit, \
    color='b', label='Fitted')

# forecasted values
ax2.plot(df_zoom.index, df_zoom.bats_forecast, \
    color='r', label='Forecast')

# set artists
ax1.set_xlabel('Date')
ax1.set_ylabel('Consumption (MW)')
ax2.grid()

plt.show()

## TBATS

# %% fit a TBATS model
tbats_estimator = TBATS(
    seasonal_periods=[24, 169],
    use_arma_errors=True,  # consider ARMA
    use_box_cox=True,  # consider Box-Cox transformation
    use_damped_trend=False, # no damping
    use_trend = False, # no trend
    n_jobs=1 # number of parallel jobs
)
tbats_model = tbats_estimator.fit(X_train)

# %% write object to file
with open('tbats_model.pkl', 'wb') as f:
    pickle.dump(tbats_model, f)

# %% load object from file
# with open('tbats_model.pkl', 'rb') as f:
#     tbats_model = pickle.load(f)

# %% print model summary
print(tbats_model.summary())

# %% extract fitted values and add to Data Frame
tbats_fit = pd.DataFrame(tbats_model.y_hat, \
    columns=['tbats_fit'], index=df.index[:len(X_train)])
df['tbats_fit'] = tbats_fit

# %% use the model to forecast
fc = tbats_model.forecast(steps=len(X_test))
fc = pd.DataFrame(fc, columns=['tbats_forecast'], \
    index=df.index[len(X_train):])
df['tbats_forecast'] = fc

# %% plot the fitted/forecasted data
plt.rcParams.update({'font.size': 14})
fig, (ax1, ax2) = plt.subplots(2, \
    figsize=(16, 10), dpi=100)

# observed data
ax1.plot(df.index, df.energy, \
    color='k', label='Actual')

# fitted values
ax1.plot(df.index, df.tbats_fit, \
    color='b', label='Fitted')

# forecasted values
ax1.plot(df.index, df.tbats_forecast, \
    color='r', label='Forecast')

# set artists
ax1.set_xlabel('Date')
ax1.set_ylabel('Consumption (MW)')
ax1.grid()
ax1.legend()

# second plot - zoom to 10 days
df_zoom = df[(df.Datetime >= '2011-04-20 00:00:00') & \
    (df.Datetime < '2011-05-10 00:00:00')]

# observed data
ax2.plot(df_zoom.index, df_zoom.energy, \
    color='k', label='Actual')

# fitted values
ax2.plot(df_zoom.index, df_zoom.tbats_fit, \
    color='b', label='Fitted')

# forecasted values
ax2.plot(df_zoom.index, df_zoom.tbats_forecast, \
    color='r', label='Forecast')

# set artists
ax1.set_xlabel('Date')
ax1.set_ylabel('Consumption (MW)')
ax2.grid()
plt.show()
# %%
