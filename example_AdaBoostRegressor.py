import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from statsmodels.tsa.stattools import kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
warnings.filterwarnings("ignore", category=InterpolationWarning)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

MODEL_NAME='AdaBoostRegressor'

# create time series data has 100 points, with trend, seasonality and noise
noise_level=0.01
trend_level=0.01
seasonality_level=1
seasonality_freq=100
len=200
ts=noise_level*np.random.randn(len)+trend_level*np.arange(0,len)+seasonality_level*np.sin(np.arange(0,len)*2*np.pi/seasonality_freq)

# plot the time series data and save it
plt.plot(ts)
plt.title('Time Series Data')
plt.savefig(f'./figures/sythetic_data_for_{MODEL_NAME}.png')
plt.close()

stationary_test=kpss(ts, regression='c', nlags="auto")
print('KPSS test statistic: ', stationary_test) 
# based on the p-value, we can reject the null hypothesis; hence the data is non-stationary

# create a differenced time series data
tsdf=pd.DataFrame({'data':ts})
print(tsdf.head(10))
# Add previous n-values
for i in range(3):
    # shift the data by i+1
    tsdf[f'data_{i+1}'] = tsdf['data'].shift(i+1)
    # difference the data
    tsdf[f'diff_{i+1}'] = tsdf['data'] - tsdf[f'data_{i+1}']

print(tsdf.head(10))
tsdf.dropna(inplace=True)


# Forecast difference of values
X, Y = tsdf[['diff_2', 'diff_3']], tsdf['diff_1']

# Split in train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=0)

# Initialize the estimator
mdl_adaboost = AdaBoostRegressor(n_estimators=50, learning_rate=0.05)

# Fit the data
mdl_adaboost.fit(X_train, Y_train)

# Make predictions
pred = mdl_adaboost.predict(X_test)

test_size = X_test.shape[0]

# plot for the diff data and original data side by side
#make the plot larger
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(list(range(test_size)), tsdf.tail(test_size).data_1  + pred, label='predicted', color='red')
plt.plot(list(range(test_size)), tsdf.tail(test_size).data, label='real', color='blue')
plt.legend(loc='best')
plt.title('Predicted vs Real with difference values')


# Forecast the original data itself
X, Y = tsdf[['data_2', 'data_3']], tsdf['data_1']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=0)
mdl_adaboost = AdaBoostRegressor(n_estimators=50, learning_rate=0.05)
mdl_adaboost.fit(X_train, Y_train)
pred = mdl_adaboost.predict(X_test)
test_size = X_test.shape[0]


plt.subplot(1,2,2)
plt.plot(list(range(test_size)),  pred, label='predicted', color='red')
plt.plot(list(range(test_size)), tsdf.tail(test_size).data, label='real', color='blue')
plt.legend(loc='best')
plt.title('Predicted vs Real with original values')
plt.savefig(f'./figures/pred_vs_real_diff_{MODEL_NAME}.png')

# Initialize a TimeSeriesSplitter
tscv = TimeSeriesSplit(n_splits=5)
# Dict to store metric value at every iteration
metric_iter = {}
X, Y = np.array(tsdf[['diff_2', 'diff_3']]), np.array(tsdf['diff_1'])
for idx, val in enumerate(tscv.split(X)):
    train_i, test_i = val
    X_train, X_test = X[train_i], X[test_i]
    Y_train, Y_test = Y[train_i], Y[test_i]
    # Initialize the estimator
    mdl_adaboost = AdaBoostRegressor(n_estimators=50, learning_rate=0.05)
    mdl_adaboost.fit(X_train, Y_train)
    pred = mdl_adaboost.predict(X_test)
    # Store metric
    metric_iter[f'iter_{idx + 1}'] = mean_absolute_error(Y_test, pred)
print(metric_iter)