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
import torch

import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME='LSTM'

# create time series data has 100 points, with trend, seasonality and noise
noise_level=0.01
trend_level=0.01
seasonality_level=1
seasonality_freq=100
LEN=400
SEQ=50
TEST_SIZE=0.2
USE_DIFF=True
ts=noise_level*np.random.randn(LEN)+trend_level*np.arange(0,LEN)+seasonality_level*np.sin(np.arange(0,LEN)*2*np.pi/seasonality_freq)

# plot the time series data and save it
plt.plot(ts)
plt.title('Time Series Data')
plt.savefig(f'./figures/sythetic_data_for_{MODEL_NAME}.png')
plt.close()

stationary_test=kpss(ts, regression='c', nlags="auto")
# print('KPSS test statistic: ', stationary_test) 
# based on the p-value, we can reject the null hypothesis; hence the data is non-stationary

# create a differenced time series data
tsdf=pd.DataFrame({'data_0':ts})
# print(tsdf.head(10))

# Add previous n-values
if not USE_DIFF:
    for i in range(SEQ):
        # shift the data by i+1
        tsdf[f'data_{i+1}'] = tsdf['data_0'].shift(i+1)
else:
    tsdf[f'data_1'] = tsdf['data_0'].shift(1)
    tsdf[f'diff_0'] = tsdf['data_0']-tsdf['data_1']
    for i in range(SEQ):
        tsdf[f'diff_{i+1}'] = tsdf[f'diff_0'].shift(i+1)
    # del column 'data_1'
    del tsdf['data_1']
    del tsdf['data_0']


print(tsdf.head(10))
tsdf.dropna(inplace=True)
# print(tsdf.describe())


X, Y = tsdf.iloc[:,1:], tsdf.iloc[:,0]

# Split in train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=False, random_state=0)
train_size=Y_train.shape[0]
test_size=Y_test.shape[0]

# transform to torch dataset
X_train = torch.from_numpy(X_train.values).float().unsqueeze(1)
Y_train = torch.from_numpy(Y_train.values).float().unsqueeze(1)
X_test = torch.from_numpy(X_test.values).float().unsqueeze(1)
Y_test = torch.from_numpy(Y_test.values).float().unsqueeze(1)
train_tensor = torch.utils.data.TensorDataset(X_train, Y_train) 
test_tensor = torch.utils.data.TensorDataset(X_test, Y_test) 
train_data_loader = torch.utils.data.DataLoader(train_tensor, batch_size=16,shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_tensor, batch_size=32,shuffle=False)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# Initialize the model
def train(model,train_loader,optimizer,crit):
    model.train()
    loss_all = 0
    for idx,data in enumerate(train_loader):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        
        optimizer.zero_grad()
        output = model(data[0]) #app_seq[batch,seq,4]  y_seq[batch*4,seq]
        # print(f'{output.shape}, {data[1].shape}')
        loss = crit(output, data[1])
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / LEN

def test(model,test_loader,crit):
    model.eval()
    loss_all=0
    with torch.no_grad():
        pred=[]
        gt=[]
        x=test_tensor[0][0].to(device) #1*SEQ, 1
        # print(x.shape)
        for idx in range(len(test_tensor)):
            y=test_tensor[idx][1].to(device)
            temp=model(x.unsqueeze(0))
            y_pred=temp.detach().cpu().numpy()
            y_gt=y.cpu().numpy()
            if len(pred)==0:
                pred=y_pred
                gt=y_gt
            else:
                if USE_DIFF:
                    pred=np.concatenate((pred,y_pred+pred[-1]),axis=0)
                    gt=np.concatenate((gt,y_gt+gt[-1]),axis=0)
                else:
                    pred=np.concatenate((pred,y_pred),axis=0)
                    gt=np.concatenate((gt,y_gt),axis=0)
                
#             print(f'test output shape: {model(data)[0].shape}; y shape: {y.shape}')
            loss = crit(temp, y.reshape(-1,1))
            loss_all=loss_all+loss.item()
            
            x[0][1:SEQ]=torch.clone(x[0][0:SEQ-1])
            x[0][0]=temp
        loss_all=loss_all/LEN

    return loss_all,pred,gt

class LSTMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # using LSTM
        hs=256
        self.lstm1=nn.LSTM(SEQ,hs,num_layers=2)
        self.fc1=nn.Linear(hs,hs)
        self.fc2=nn.Linear(hs,1)
        # dropout layer
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # print(x.shape) # 16, 1, 5
        # reverse order on the last dim of x
        x=torch.flip(x,[2])
        #pass x through lstm layer
        x,_=self.lstm1(x) #16,1,hs
        # x=self.dropout(x)
        # use all the hidden states
        x=F.relu(self.fc1(x[:,-1,:]))#16,hs 
        x=self.dropout(x)
        x=self.fc2(x)
        # print(x.shape)
        return x


model=LSTMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
# initialize the loss function using mean squared error
crit=nn.MSELoss()
history=[]
test_history=[]
best_validate_mae = np.inf
validate_score_non_decrease_count = 0
best_epoch=0
for epoch in range(50):
    loss=train(model,train_data_loader,optimizer,crit)
    history.append(loss)
#      progress monitor:
    if (epoch+1) % 10 ==0:
        print(f'{epoch:3d} -- train loss: {loss:.8f}')
        

test_loss,pred,gt=test(model,test_data_loader,crit)
print(f'test loss: {test_loss:.8f}')

# plot
plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
plt.plot(pred, label='predicted', color='red')
plt.plot(gt, label='real', color='blue',alpha=0.5)
plt.legend(loc='best')
plt.title(f'{MODEL_NAME},SEQ={SEQ},train loss:{loss:.4f}, test loss: {test_loss:.4f}')
plt.savefig(f'./figures/pred_vs_real_diff_{MODEL_NAME}.png')