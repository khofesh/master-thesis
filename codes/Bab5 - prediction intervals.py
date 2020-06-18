
# coding: utf-8

# In[1]:


import tseriesRoutines as routines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import ceil, sqrt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras import regularizers
from keras.constraints import max_norm, non_neg, unit_norm, min_max_norm
from keras.wrappers.scikit_learn import KerasRegressor


##########################################################################################
# RESULT REPRODUCIBILITY                                                                 #
##########################################################################################
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(42)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(42)
##########################################################################################


# In[2]:


# DATA UNTUK TRAINING
# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()


# In[17]:


# DATA UNTUK VALIDASI
# make connection to sqlite db
conn2 = sqlite3.connect('validasi.db')
c2 = conn2.cursor()

# enable foreign keys
c2.execute("PRAGMA foreign_keys = ON")
conn2.commit()


# In[18]:


# import tseriesNN.py
import tseriesNN as tnn
# import validasiModelForecasting.py
import validasiModelForecasting as vmf


# In[5]:


# Training set
product = tnn.genData('5aa2ad7735d6d34b0032a795', conn, c, impute=False, freq='daily')
X_train, y_train, X_test, y_test, dftrain, scaler = tnn.splitDataNN(product, percent=0.2, n_in=2, n_out=2)


# In[6]:


from sklearn.utils import resample


# In[20]:


# Validation set
mongoid = vmf.getTheRightID('5aa2ad7735d6d34b0032a795')
productval = routines.genDataVal(mongoid, conn2, c2, impute=False, freq='daily')
X_val, y_val, dftrainval, scalerval = vmf.splitDataNNVal(productval, n_in=2, n_out=2)


# In[9]:


yboot = pd.DataFrame()
for i in range(0, 250):
    X_samples, y_samples = resample(X_train, y_train)

    model2 = tnn.lstmModel(X_samples, y_samples, X_test, y_test, epochs=200, batch_size=8, 
                           units=4, drop=0.002, recdrop=0.002, plot=False)

    # make prediction
    ypred = model2.model.predict(X_val)
    # reshape X
    X = X_val.reshape((X_val.shape[0], X_val.shape[2]))
    # invert scaling predicted data
    inv_ypred = np.concatenate((X[:, :], ypred), axis=1)
    inv_ypred = scaler.inverse_transform(inv_ypred)
    inv_ypred = inv_ypred[:, -1]
    
    print('iterasi {0}'.format(i))
    
    yboot[i] = pd.Series(inv_ypred)
    # save memory
    del X_samples, y_samples, model2, X, ypred, inv_ypred


# In[ ]:


yboot.to_csv('./csvfiles/ybootstrap_bab5.csv', index=False)


# In[2]:


yboot = pd.read_csv('./csvfiles/ybootstrap_bab5.csv')


# In[4]:


yboot.shape


# In[5]:


yboot.loc[0,:].sum()/100


# In[13]:


# khosravi et al. halaman 6, formula (28)
yboot['yhat'] = yboot.sum(axis=1)/250


# In[12]:


# khosravi et al. halaman 6, formula (29)
listOfVar = []
for i in range(len(yboot)):
    variance = np.square((yboot.iloc[i, :-1] - yboot.iloc[i, 250]).sum())/(250-1)
    listOfVar.append(variance)


# In[16]:




