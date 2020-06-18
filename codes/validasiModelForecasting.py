'''
Validasi Model Forecasting

pilihan:
    2 produk di database product.db dgn actualrevcount > 900:
        - 5aa2ad7735d6d34b0032a795
        - 5aa39533ae1f941be7165ecd
    cluster 1
        - 5aa2c35e35d6d34b0032a796
    cluster 2
        - 5a93e8768cbad97881597597 
    cluster 3
        - 5a9347b98cbad97074cb1890

    'MI' = multivariate imputed
    'UI' = univariate imputed
    'MN' = multivariate notimputed
    'UN' = univariate notimputed

'''
# library buatan sendiri
import tseriesRoutines as routines
# routines
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sqlite3
from math import ceil, sqrt
import os
# scikit-learn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
# SVR
from sklearn.svm import SVR
# Linear Regression
from sklearn.linear_model import LinearRegression
# Keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras import regularizers
from keras.constraints import max_norm, non_neg, unit_norm, min_max_norm
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model

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

def splitDataNNVal(df, n_in=1, n_out=1, scale=True):
    '''
    unlike routines.splitDataNNVal, this splitDataNNVal doesn't split data 
    into train and test set data
    '''
    dftrain = routines.series_to_supervised(df, n_in=n_in, n_out=n_out)
    # specific to this case
    dftrain = dftrain.drop(dftrain.columns[[4, 5]], axis=1)
    values = dftrain.values

    if scale:
        scaler = MinMaxScaler()
        values = scaler.fit_transform(values)
    else:
        pass

    # training data
    X, y = values[:, :-1], values[:, -1]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X, y, dftrain, scaler

def splitDataNNValUni(df, n_in=1, n_out=1, scale=True):
    '''
    for Univariate NN forecasting model

    df: pandas dataframe. 3 columns (sales, rating, ovsentiment) with date as index
    n_in:
    n_out:
    scale:
    percent:
    X_train, y_train, X_test, y_test, dftrain = splitDataNN(product, n_in=1,
        n_out=1, scale=True, percent=0.2)
    '''
    dftrain = routines.series_to_supervised(df, n_in=n_in, n_out=n_out)
    # specific to this case
    dftrain = dftrain.drop(dftrain.columns[[4, 5]], axis=1)
    dftrain = dftrain[['var1(t-1)', 'var1(t)']]
    values = dftrain.values

    if scale:
        scaler = MinMaxScaler()
        values = scaler.fit_transform(values)
    else:
        pass

    # training data
    X, y = values[:, :-1], values[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X, y, dftrain, scaler

def splitDataSVRVal(df, n_in=1, n_out=1, scale=True):
    '''
    for Multivariate SVR/SLR forecasting model

    df: pandas dataframe. 3 columns (sales, rating, ovsentiment) with date as index
    n_in:
    n_out:
    scale:
    percent:
    X_train, y_train, X_test, y_test, dftrain = splitDataNN(product, n_in=1,
        n_out=1, scale=True, percent=0.2)
    '''
    dftrain = routines.series_to_supervised(df, n_in=n_in, n_out=n_out)
    # specific to this case
    dftrain = dftrain.drop(dftrain.columns[[4, 5]], axis=1)
    values = dftrain.values

    if scale:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    else:
        pass

    # validation data
    X, y = values[:, :-1], values[:, -1]
    return X, y, dftrain, scaler

def splitDataSVRValUni(df, n_in=1, n_out=1, scale=True):
    '''
    for Univariate SVR/SLR forecasting model

    df: pandas dataframe. 3 columns (sales, rating, ovsentiment) with date as index
    n_in:
    n_out:
    scale:
    percent:
    X_train, y_train, X_test, y_test, dftrain = splitDataNN(product, n_in=1,
        n_out=1, scale=True, percent=0.2)
    '''
    dftrain = routines.series_to_supervised(df, n_in=n_in, n_out=n_out)
    # specific to this case
    dftrain = dftrain.drop(dftrain.columns[[4, 5]], axis=1)
    dftrain = dftrain[['var1(t-1)', 'var1(t)']]
    values = dftrain.values

    if scale:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    else:
        pass

    # training data
    X, y = values[:, :-1], values[:, -1]
    return X, y, dftrain, scaler

def evalForecast(model, X, y, inverse=False, scaler=None):
    '''
    Evaluate time series forecasting model
    '''
    if inverse and scaler:
        # make prediction
        ypred = model.predict(X)
        # invert scaling predicted data
        inv_ypred = np.concatenate((X[:, :], ypred.reshape((-1,1))), axis=1)
        inv_ypred = scaler.inverse_transform(inv_ypred)
        inv_ypred = inv_ypred[:, -1]
        # invert scaling for actual data
        y = y.reshape((len(y), 1))
        inv_y = np.concatenate((X[:, :], y.reshape((-1,1))), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -1]
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_pred=inv_ypred, y_true=inv_y))
        # MAE
        mae = mean_absolute_error(y_pred=inv_ypred, y_true=inv_y)
    else:
        # make prediction
        ypred = model.model.predict(X)
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_pred=ypred, y_true=y))
        # MAE
        mae = mean_absolute_error(y_pred=ypred, y_true=y)

    print('Validasi RMSE: {0:.5f}'.format(rmse))
    print('Validasi MAE: {0:.5f}'.format(mae))

def evalForecastNN(model, X, y, inverse=False, scaler=None):
    '''
    Evaluate time series forecasting model
    '''
    if inverse and scaler:
        # make prediction
        ypred = model.predict(X)
        # reshape X
        X = X.reshape((X.shape[0], X.shape[2]))
        # invert scaling predicted data
        inv_ypred = np.concatenate((X[:, :], ypred), axis=1)
        inv_ypred = scaler.inverse_transform(inv_ypred)
        inv_ypred = inv_ypred[:, -1]
        # invert scaling for actual data
        y = y.reshape((len(y), 1))
        inv_y = np.concatenate((X[:, :], y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -1]
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_pred=inv_ypred, y_true=inv_y))
        # MAE
        mae = mean_absolute_error(y_pred=inv_ypred, y_true=inv_y)
    else:
        # make prediction
        ypred = model.predict(X)
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_pred=ypred, y_true=y))
        # MAE
        mae = mean_absolute_error(y_pred=ypred, y_true=y)

    print('Validasi RMSE: {0:.5f}'.format(rmse))
    print('Validasi MAE: {0:.5f}'.format(mae))


def getTheRightModel(modelsname, dirpath):
    '''
    returns mongoid and model.
    modelsname: string. 
        model's name example = 'forecast_svr_5aa2c35e35d6d34b0032a796_MN_lag1.pkl'
    dirpath: directory path.
    '''
    listOfChar = modelsname.split('_')
    mongoid = listOfChar[2]
    if listOfChar[-1].split('.')[-1] == 'h5':
        model = load_model(dirpath + modelsname)
        kind = listOfChar[-2]
    elif listOfChar[-1].split('.')[-1] == 'pkl':
        model = joblib.load(dirpath + modelsname)
        kind = listOfChar[1]
    else:
        print('Not Supported!')

    return mongoid, model, kind

def getTheRightID(mongoid):
    #'5aa2ad7735d6d34b0032a795' -> '5b32126eb8a9f70142f6a766'
    #'5aa39533ae1f941be7165ecd' -> '5b322e9cb8a9f70142f6a767'
    #'5aa2c35e35d6d34b0032a796' -> '5b3277a3b8a9f70142f6a76b'
    #'5a93e8768cbad97881597597' -> '5b324265b8a9f70142f6a768'
    #'5a9347b98cbad97074cb1890' -> '5b38f8bd35d6d30034572e0e'
    if mongoid == '5aa2ad7735d6d34b0032a795':
        mongoid2 = '5b32126eb8a9f70142f6a766'
    elif mongoid == '5aa39533ae1f941be7165ecd':
        mongoid2 = '5b322e9cb8a9f70142f6a767'
    elif mongoid == '5aa2c35e35d6d34b0032a796':
        mongoid2 = '5b3277a3b8a9f70142f6a76b'
    elif mongoid == '5a93e8768cbad97881597597':
        mongoid2 = '5b324265b8a9f70142f6a768'
    elif mongoid == '5a9347b98cbad97074cb1890':
        mongoid2 = '5b38f8bd35d6d30034572e0e'
    else:
        print('Not Supported!')

    return mongoid2

def modelOnVal(conn, c, lag=1, datatype='MI'):
    '''
    measure chosen model on val set
    conn: sqlite3 connection
    c: sqlite3 cursor
    lag: int. value ranges from 1-5.
    datatype: string. 'MI' = multivariate imputed
                      'UI' = univariate imputed
                      'MN' = multivariate notimputed
                      'UN' = univariate notimputed
    output:
        ID
        model
        performance
    '''
    if lag == 1:
        dirpath1 = './training/lag1/'
    elif lag == 2:
        dirpath1 = './training/lag2/'
    elif lag == 3:
        dirpath1 = './training/lag3/'
    elif lag == 4:
        dirpath1 = './training/lag4/'
    elif lag == 5:
        dirpath1 = './training/lag5/'
    else:
        print('What The Fuck, It\'s Not Supported?!!')

    if datatype == 'MI':
        dirpath2 = 'MI/'
        impute = True
    elif datatype == 'UI':
        dirpath2 = 'UI/'
        impute = True
    elif datatype == 'MN':
        dirpath2 = 'MN/'
        impute = False
    elif datatype == 'UN':
        dirpath2 = 'UN/'
        impute = False
    else:
        print('Not Supported!')

    # get directory path of models
    dirpath = dirpath1 + dirpath2
    listOfModels = os.listdir(dirpath)

    print(dirpath)
    for i in listOfModels:
        mongoid1, model, kind = getTheRightModel(i, dirpath)
        mongoid = getTheRightID(mongoid1)
        if kind in ['lstmmodel', 'grumodel', 'flatmodel'] and datatype in ['MN', 'MI']:
            # if kind is multivariate neural network model whether imputed or not
            product = routines.genDataVal(mongoid, conn, c, impute=impute, freq='daily')
            X_val, y_val, dftrain, scaler = splitDataNNVal(product, n_in=lag, n_out=lag)

            print('id: {0}, {1}, {2}, {3}'.format(mongoid1, kind, datatype, lag))
            evalForecastNN(model, X_val, y_val, inverse=True, scaler=scaler)

        elif kind in ['lstmmodel', 'grumodel', 'flatmodel'] and datatype in ['UN', 'UI']:
            # if kind is univariate neural network model whether imputed or not
            product = routines.genDataVal(mongoid, conn, c, impute=impute, freq='daily')
            X_val, y_val, dftrain, scaler = splitDataNNValUni(product, n_in=lag, n_out=lag)

            print('id: {0}, {1}, {2}, {3}'.format(mongoid1, kind, datatype, lag))
            evalForecastNN(model, X_val, y_val, inverse=True, scaler=scaler)

        elif kind == 'svr' and datatype in ['MN', 'MI']:
            # if kind is multivariate SVM model whether imputed or not
            product = routines.genDataVal(mongoid, conn, c, impute=impute, freq='daily')
            X_val, y_val, dftrain, scaler = splitDataSVRVal(product, n_in=lag, n_out=lag)

            print('id: {0}, {1}, {2}, {3}'.format(mongoid1, kind, datatype, lag))
            evalForecast(model, X_val, y_val, inverse=True, scaler=scaler)

        elif kind == 'svr' and datatype in ['UN', 'UI']:
            # if kind is multivariate SVM model whether imputed or not
            product = routines.genDataVal(mongoid, conn, c, impute=impute, freq='daily')
            X_val, y_val, dftrain, scaler = splitDataSVRValUni(product, n_in=lag, n_out=lag)

            print('id: {0}, {1}, {2}, {3}'.format(mongoid1, kind, datatype, lag))
            evalForecast(model, X_val, y_val, inverse=True, scaler=scaler)

        elif kind == 'LR' and datatype in ['MN', 'MI']:
            # if kind is multivariate SVM model whether imputed or not
            product = routines.genDataVal(mongoid, conn, c, impute=impute, freq='daily')
            X_val, y_val, dftrain, scaler = splitDataSVRVal(product, n_in=lag, n_out=lag)

            print('id: {0}, {1}, {2}, {3}'.format(mongoid1, kind, datatype, lag))
            evalForecast(model, X_val, y_val, inverse=True, scaler=scaler)

        elif kind == 'LR' and datatype in ['UN', 'UI']:
            # if kind is multivariate SVM model whether imputed or not
            product = routines.genDataVal(mongoid, conn, c, impute=impute, freq='daily')
            X_val, y_val, dftrain, scaler = splitDataSVRValUni(product, n_in=lag, n_out=lag)

            print('id: {0}, {1}, {2}, {3}'.format(mongoid1, kind, datatype, lag))
            evalForecast(model, X_val, y_val, inverse=True, scaler=scaler)
        else:
            print('Not Supported!!')

# make connection to sqlite db
conn = sqlite3.connect('validasi.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

#'MI' = multivariate imputed
#'UI' = univariate imputed
#'MN' = multivariate notimputed
#'UN' = univariate notimputed

#modelOnVal(conn, c, lag=5, datatype='UN')
