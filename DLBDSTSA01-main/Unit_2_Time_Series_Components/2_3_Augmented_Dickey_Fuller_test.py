# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Augmented Dickey-Fuller Test

# %% import modules
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# %% stock selection
stock = ['DBK.DE'] # Deutsche Bank stock code
stock_df = yf.download(stock, 
                      start='2019-01-01', 
                      end='2020-12-31', 
                      progress=False)

# %% plot the data
plt.figure(figsize=(16,5), dpi=100)
plt.plot(stock_df.index, stock_df['Close']) 
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.title('Deutsche Bank - Stock Prices')

# %% plot ACF’s
fig, axes = plt.subplots(2, 1, \
    figsize=(16,10), dpi=100)

# ACF closing price
plot_acf(stock_df.Close, alpha=0.05, ax=axes[0], \
    title='Autocorrelation - Closing Prices')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation')

# ACF difference
plot_acf(stock_df.Close.diff().dropna(), \
    alpha=0.05, ax=axes[1], \
    title='Autocorrelation -\ Differenced Prices')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')

# %% conduct augmented Dickey-Fuller test
resultADF = adfuller(stock_df.Close, \
    regression="n", regresults=True)

# %% print results
print('ADF Statistic: %f' % resultADF[0])
print('p-value: %f' % resultADF[1])
for key, value in resultADF[2].items():
	print('\t%s: %.3f' % (key, value))

# console output:
# ADF Statistic: 0.008459
# p-value: 0.687086
# 	1%: -2.570
# 	5%: -1.942
# 	10%: -1.616

# %% conduct augmented Dickey-Fuller test
# on differenced series
resultADF = adfuller(stock_df.Close.diff().dropna(), \
    regression="n", regresults=True)

print('ADF Statistic: %f' % resultADF[0])
print('p-value: %f' % resultADF[1])
for key, value in resultADF[2].items():
	print('\t%s: %.3f' % (key, value))

# %%
