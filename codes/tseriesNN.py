'''
Train Forecasting Model with RNN
'''

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

def genData(mongoid, conn, cursor, impute=True, freq='daily'):
    '''
    Generate a timeseries dataframe for timeseries modelling.
    mongoid: str. string of mongodb id.
    conn: sqlite3 connection.
    cursor: sqlite3 cursor.
    impute:
    freq:
    actualrevcount:
    '''
    initial = routines.sqlToDf(conn, cursor)
    allproduct = initial.selectReview3(mongoid, impute=impute)
    product = routines.tsSalesRateSentiment(allproduct, freq=freq)
    return product
    # product = genData('5aa2ad7735d6d34b0032a795', conn, c, impute=True,
    #   freq='daily')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data.copy()
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg = agg.dropna()
    return agg

def splitDataNN(df, n_in=1, n_out=1, scale=True, percent=0.2):
    '''
    df: pandas dataframe. 3 columns (sales, rating, ovsentiment) with date as index
    n_in:
    n_out:
    scale:
    percent:
    X_train, y_train, X_test, y_test, dftrain = splitDataNN(product, n_in=1, 
        n_out=1, scale=True, percent=0.2)
    '''
    dftrain = series_to_supervised(df, n_in=n_in, n_out=n_out)
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
    # train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent, 
            shuffle=False, random_state=42)
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train, y_train, X_test, y_test, dftrain, scaler

# computing the common-sense baseline MAE
def evaluate_naive_method(val_steps, val_gen):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

def flatModel(xtrain, ytrain, xtest, ytest, epochs, batch_size, units=50, shuffle=False, 
        metrics=['accuracy'], loss='mae', plot=True, lr=0.001, accplot=False):
    '''
    xtrain:
    ytrain:
    epochs:
    batch_size:
    units:
    shuffle:
    metrics:
    loss:
    plot:
    flatmodel = flatModel(X_train, y_train, epochs=80, batch_size=64)
    '''
    # 1 - densely connected model
    #############################
    model = Sequential()
    model.add(layers.Flatten(input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(lr=lr), loss=loss, metrics=metrics)
    history = model.fit(xtrain, ytrain, 
                        epochs=epochs, batch_size=batch_size, 
                        validation_data=(xtest, ytest),
                        verbose=0, shuffle=False)
    # save the output
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(loss) + 1)

    # plot
    if plot and accplot:
        # loss plot
        plt.figure(1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        # accuracy plot
        plt.figure(2)
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Training and validation Accuracy')
        plt.legend()
        plt.show()
        # print results
        print('loss: {0}'.format(np.mean(loss)))
        print('val_loss: {0}'.format(np.mean(val_loss)))
        print('acc: {0}'.format(np.mean(acc)))
        print('val_acc: {0}'.format(np.mean(val_acc)))
    elif plot:
        # loss plot
        plt.figure(1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        # print results
        print('loss: {0}'.format(np.mean(loss)))
        print('val_loss: {0}'.format(np.mean(val_loss)))
    else:
        pass

    plt.clf()
    # return model
    return history

def gruDropModel(xtrain, ytrain, xtest, ytest, epochs, batch_size, units, drop, recdrop, 
        shuffle=False, metrics=['accuracy'], loss='mae', plot=True, lr=0.001, accplot=False):
    '''
    xtrain:
    ytrain:
    epochs:
    batch_size:
    units:
    shuffle:
    metrics:
    loss:
    plot:
    grudropmodel = gruDropModel(X_train, y_train, epochs=1000, batch_size=64, units=256, 
        drop=0.2, recdrop=0.2)
    '''
    # 3 - GRU-based model with dropout
    ##################################
    model = Sequential()
    model.add(layers.GRU(units, 
        dropout=drop,
        recurrent_dropout=recdrop,
        input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(lr=lr), loss=loss, metrics=metrics)
    history = model.fit(xtrain, ytrain, 
                        epochs=epochs, batch_size=batch_size, 
                        validation_data=(xtest, ytest),
                        verbose=0, shuffle=False)
    # save the output
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(loss) + 1)

    # plot
    if plot and accplot:
        # loss plot
        plt.figure(1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        # accuracy plot
        plt.figure(2)
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Training and validation Accuracy')
        plt.legend()
        plt.show()
        # print results
        print('loss: {0}'.format(np.mean(loss)))
        print('val_loss: {0}'.format(np.mean(val_loss)))
        print('acc: {0}'.format(np.mean(acc)))
        print('val_acc: {0}'.format(np.mean(val_acc)))
    elif plot:
        # loss plot
        plt.figure(1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        # print results
        print('loss: {0}'.format(np.mean(loss)))
        print('val_loss: {0}'.format(np.mean(val_loss)))
    else:
        pass

    plt.clf()
    # return model
    return history

def lstmModel(xtrain, ytrain, xtest, ytest, epochs, batch_size, units, drop, recdrop, 
        shuffle=False, metrics=['accuracy'], loss='mae', plot=True, lr=0.001, accplot=False):
    '''
    xtrain:
    ytrain:
    epochs:
    batch_size:
    units:
    shuffle:
    metrics:
    loss:
    plot:
    lstmmodel = lstmModel(X_train, y_train, epochs=500, batch_size=64, units=256, 
        drop=0.2, recdrop=0.2)
    '''
    # 4 - LSTM-based model
    ######################
    model = Sequential()
    model.add(layers.LSTM(units, dropout=drop, recurrent_dropout=recdrop,
        input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(lr=lr), loss=loss, metrics=metrics)
    history = model.fit(xtrain, ytrain, 
                        epochs=epochs, batch_size=batch_size, 
                        validation_data=(xtest, ytest),
                        verbose=0, shuffle=False)
    # save the output
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(loss) + 1)

    # plot
    if plot and accplot:
        # loss plot
        plt.figure(1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        # accuracy plot
        plt.figure(2)
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Training and validation Accuracy')
        plt.legend()
        plt.show()
        # print results
        print('loss: {0}'.format(np.mean(loss)))
        print('val_loss: {0}'.format(np.mean(val_loss)))
        print('acc: {0}'.format(np.mean(acc)))
        print('val_acc: {0}'.format(np.mean(val_acc)))
    elif plot:
        # loss plot
        plt.figure(1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        # print results
        print('loss: {0}'.format(np.mean(loss)))
        print('val_loss: {0}'.format(np.mean(val_loss)))
    else:
        pass

    plt.clf()
    # return model
    return history

def evalForecast(model, X, y, inverse=False, scaler=None):
    '''
    Evaluate time series forecasting model
    '''
    if inverse and scaler:
        # make prediction
        ypred = model.model.predict(X)
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
        ypred = model.model.predict(X)
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_pred=ypred, y_true=y))
        # MAE
        mae = mean_absolute_error(y_pred=ypred, y_true=y)

    print('Validasi RMSE: {0:.5f}'.format(rmse))
    print('Validasi MAE: {0:.5f}'.format(mae))

def makeModelName(mongoid, algo='nn', impute=True, multivariate=True, kind='flatmodel', lag=1):
    '''
    Name a model.
    algo: nn= neural network
    mongoid: id
    impute: impute or note. boolean
    lag: time lag.

    format:
    forecast_algo_id_MI_kind_lag.h5
    forecast_nn_5aa2ad7735d6d34b0032a795_flatmodel_notimputed.h5'
    '''
    one = 'forecast_'
    two = algo + '_'
    three = mongoid + '_'
    if impute:
        four2 = 'I'
    else:
        four2 = 'N'

    if multivariate:
        four1 = 'M'
    else:
        four1 = 'U'

    four = four1 + four2 + '_'
    five = kind + '_'
    six = 'lag' + str(lag) + '.h5'
    
    name = one + two + three + four + five + six
    return name

def dirLagPath(lag):
    '''
    Returns directory path to save model
    lag: integer. 1-5
    if lag = 1
        return './training/lag1/'
    '''
    if lag == 1:
        return './training/lag1/'
    elif lag == 2:
        return './training/lag2/'
    elif lag == 3:
        return './training/lag3/'
    elif lag == 4:
        return './training/lag4/'
    elif lag == 5:
        return './training/lag5/'
    else:
        print('Not Supported! lag 1 - 5 Only')

def makeModel(mongoid, impute=True, inverse=False, save=False, lag=1):
    listOfIds = ['5aa2ad7735d6d34b0032a795', '5aa39533ae1f941be7165ecd',
            '5aa2c35e35d6d34b0032a796', '5a93e8768cbad97881597597',
            '5a9347b98cbad97074cb1890']

    print('#############')
    if impute:
        print('imputed')
    else:
        print('not imputed')
    print('Lag: {0}'.format(lag))
    print('#############')

    # Directory path
    dirpath = dirLagPath(lag=lag)

    if mongoid == listOfIds[0]:
        product = genData('5aa2ad7735d6d34b0032a795', conn, c, impute=impute, freq='daily')
        X_train, y_train, X_test, y_test, dftrain, scaler = splitDataNN(product, percent=0.2, n_in=lag, n_out=lag)
        print(mongoid)
        print('Densely Connected Model')
        flatmodel = flatModel(X_train, y_train, X_test, y_test, epochs=80, batch_size=8, 
                loss='mae', units=4)
        evalForecast(flatmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        flatname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='flatmodel', lag=lag)
        print('RNN - LSTM')
        lstmmodel = lstmModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=8, 
                units=4, drop=0.002, recdrop=0.002)
        evalForecast(lstmmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        lstmname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='lstmmodel', lag=lag)
        print('RNN - GRU')
        grudropmodel = gruDropModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=8, 
                units=8, drop=0.01, recdrop=0.01, lr=0.0001)
        evalForecast(grudropmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        gruname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='grumodel', lag=lag)


        if impute and save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        elif save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        else:
            pass

    elif mongoid == listOfIds[1]:
        product = genData('5aa39533ae1f941be7165ecd', conn, c, impute=impute, freq='daily')
        X_train, y_train, X_test, y_test, dftrain, scaler = splitDataNN(product, percent=0.2, n_in=lag, n_out=lag)
        print(mongoid)
        print('Densely Connected Model')
        flatmodel = flatModel(X_train, y_train, X_test, y_test, epochs=80, batch_size=16, units=8)
        evalForecast(flatmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        flatname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='flatmodel', lag=lag)
        print('RNN - LSTM')
        lstmmodel = lstmModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, units=8, drop=0.01, recdrop=0.01, lr=0.0005)
        evalForecast(lstmmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        lstmname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='lstmmodel', lag=lag)
        print('RNN - GRU')
        grudropmodel = gruDropModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, units=8, drop=0.01, recdrop=0.01, lr=0.0009)
        evalForecast(grudropmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        gruname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='grumodel', lag=lag)


        if impute and save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        elif save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        else:
            pass

    elif mongoid == listOfIds[2]:
        product = genData('5aa2c35e35d6d34b0032a796', conn, c, impute=impute, freq='daily')
        X_train, y_train, X_test, y_test, dftrain, scaler = splitDataNN(product, percent=0.2, n_in=lag, n_out=lag)
        print(mongoid)
        print('Densely Connected Model')
        flatmodel = flatModel(X_train, y_train, X_test, y_test, epochs=80, batch_size=16, loss='mae', units=8)
        evalForecast(flatmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        flatname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='flatmodel', lag=lag)
        print('RNN - LSTM')
        lstmmodel = lstmModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=8, units=4, drop=0.01, recdrop=0.01, lr=0.00008)
        evalForecast(lstmmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        lstmname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='lstmmodel', lag=lag)
        print('RNN - GRU')
        grudropmodel = gruDropModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=16, units=8, drop=0.01, recdrop=0.01, lr=0.0005)
        evalForecast(grudropmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        gruname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='grumodel', lag=lag)

        if impute and save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        elif save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        else:
            pass

    elif mongoid == listOfIds[3]:
        product = genData('5a93e8768cbad97881597597', conn, c, impute=impute, freq='daily')
        X_train, y_train, X_test, y_test, dftrain, scaler = splitDataNN(product, percent=0.2, n_in=lag, n_out=lag)
        print(mongoid)
        print('Densely Connected Model')
        flatmodel = flatModel(X_train, y_train, X_test, y_test, epochs=80, batch_size=64, loss='mae', units=8)
        evalForecast(flatmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        flatname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='flatmodel', lag=lag)
        print('RNN - LSTM')
        lstmmodel = lstmModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=16, units=8, drop=0.02, recdrop=0.02, lr=0.0008)
        evalForecast(lstmmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        lstmname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='lstmmodel', lag=lag)
        print('RNN - GRU')
        grudropmodel = gruDropModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=16, units=8, drop=0.01, recdrop=0.01, lr=0.0008)
        evalForecast(grudropmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        gruname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='grumodel', lag=lag)

        if impute and save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        elif save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        else:
            pass

    elif mongoid == listOfIds[4]:
        product = genData('5a9347b98cbad97074cb1890', conn, c, impute=impute, freq='daily')
        X_train, y_train, X_test, y_test, dftrain, scaler = splitDataNN(product, percent=0.2, n_in=lag, n_out=lag)
        print(mongoid)
        print('Densely Connected Model')
        flatmodel = flatModel(X_train, y_train, X_test, y_test, epochs=80, batch_size=16, loss='mae', units=8)
        evalForecast(flatmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        flatname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='flatmodel', lag=lag)
        print('RNN - LSTM')
        lstmmodel = lstmModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=8, units=4, drop=0.02, recdrop=0.02, lr=0.001)
        evalForecast(lstmmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        lstmname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='lstmmodel', lag=lag)
        print('RNN - GRU')
        grudropmodel = gruDropModel(X_train, y_train, X_test, y_test, epochs=200, batch_size=8, units=4, drop=0.05, recdrop=0.05, lr=0.001)
        evalForecast(grudropmodel, X_train, y_train, inverse=inverse, scaler=scaler)
        gruname = makeModelName(mongoid, algo='nn', impute=impute, multivariate=True,
                kind='grumodel', lag=lag)

        if impute and save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        elif save:
            flatmodel.model.save(dirpath + flatname)
            lstmmodel.model.save(dirpath + lstmname)
            grudropmodel.model.save(dirpath + gruname)
        else:
            pass

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

listOfIds = ['5aa2ad7735d6d34b0032a795', '5aa39533ae1f941be7165ecd',
        '5aa2c35e35d6d34b0032a796', '5a93e8768cbad97881597597',
        '5a9347b98cbad97074cb1890']

#'MI' = multivariate imputed
#'UI' = univariate imputed
#'MN' = multivariate notimputed
#'UN' = univariate notimputed
# lag -> 1, 2, 3, 4, 5

# Multivariate-Imputed
for i in listOfIds:
    makeModel(i, impute=True, inverse=True, save=False, lag=2)

# Multiviarate-Not Imputed
#for i in listOfIds:
#    makeModel(i, impute=False, inverse=True, save=True, lag=5)

