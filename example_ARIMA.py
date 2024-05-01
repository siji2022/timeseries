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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model  import ARIMA

MODEL_NAME='ARIMA'

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
X, Y = tsdf[['data_2', 'data_3']], tsdf['data_1']

# Split in train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=0)
train_size=Y_train.shape[0]
test_size=Y_test.shape[0]
# Initialize the estimator
ARMAmodel = ARIMA(Y_train, order = (1, 1, 1))
ARMAmodel = ARMAmodel.fit()
print(ARMAmodel.summary())
y_pred = ARMAmodel.forecast(100)


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(y_pred, label='predicted', color='red')
plt.plot(Y_train, label='train', color='teal',alpha=0.5)
plt.plot(Y, label='real', color='blue',alpha=0.5)
plt.legend(loc='best')
plt.title(f'Predicted vs Real {MODEL_NAME}')
plt.savefig(f'./figures/pred_vs_real_diff_{MODEL_NAME}.png')