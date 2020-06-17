#!/usr/bin/env python
# coding: utf-8

# In[13]:


# this function plots rolling mean and standard deviations against originla dataset
# in the second section, the most relevant info from the Dickey-fuller test are displayed
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from statsmodels.tsa.stattools import adfuller
def stationarity_check(TS):
    
   
    # Calculate rolling statistics with a week window (168 hours)
    roll_mean = TS.rolling(window=168, center=False).mean()
    roll_std = TS.rolling(window=168, center=False).std()
    
    # Perform the Dickey Fuller test
    dftest = adfuller(TS) 
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(TS, color='blue',label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print('Results of Dickey-Fuller Test: \n')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                             '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    return None


# In[14]:


# this functions prepares series to be fed into an RNN.
# it defines the number of variables and number of time stamps for each variables 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[15]:


# plots the loss function graph based on mean squared error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# In[16]:


# this function makes predictions with the trained RNN, 
# reconverts all normalised results and compares them against the observations
# prints out MSE and graphs results (observation/predictions) for the first n timesteps
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from joblib import dump, load
scaler = load('scaler_training') 
import numpy as np
from sklearn.metrics import mean_squared_error
def mse_plot_pred (model, data_x, data_y, timesteps):
    
    # make a prediction
    yhat = model.predict(data_x)
    data_x = data_x.reshape((data_x.shape[0], 24))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, data_x[:, -23:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    data_y = data_y.reshape((len(data_y), 1))
    inv_y = np.concatenate((data_y, data_x[:, -23:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    

    fig = plt.subplots(figsize=(20,10))
    aa=[x for x in range(timesteps)]
    plt.title('Sales per hour', fontsize= 20)
    plt.plot(aa, inv_y[:timesteps], marker='.', label="actual")
    plt.plot(aa, inv_yhat[:timesteps], 'r', label="prediction")
    plt.ylabel('Sales', size=20)
    plt.xlabel('Time step', size=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
    rmse = (mean_squared_error(inv_y, inv_yhat))
    print('MSE: %.3f' % rmse)


# In[ ]:





# In[ ]:





# In[ ]:




