import tseriesRoutines as routines
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sqlite3
# SVR
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

# RESULT REPRODUCIBILITY
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

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
    np.random.seed(42)
    initial = routines.sqlToDf(conn, cursor)
    allproduct = initial.selectReview3(mongoid, impute=impute)
    product = routines.tsSalesRateSentiment(allproduct, freq=freq)
    return product
    # product = genData('5aa2ad7735d6d34b0032a795', conn, c, impute=True, 
    #   freq='daily')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
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

#Adj R square
def adj_r2_score(model,X_test, y_test,):
    y_pred = model.predict(X_test)
    # model.coefs_ doesn't exist
    adj = 1 - float(len(y_test)-1)/(len(y_test)-model.n_features_-1) * \
            (1 - r2_score(y_test,y_pred))
    return adj

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
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    else:
        pass

    # training data
    X, y = values[:, :-1], values[:, -1]
    # train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent,
            shuffle=False, random_state=42)
    return X_train, y_train, X_test, y_test, dftrain, scaler

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
        inv_y = np.concatenate((X[:, :], y.reshape((-1,1))), axis=1)
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

def makeModelName(mongoid, algo='svr', impute=True, multivariate=True, lag=1):
    '''
    Name a model.
    algo: nn= neural network
    mongoid: id
    impute: impute or note. boolean
    lag: time lag.

    format:
    forecast_algo_id_MI_lag.h5
    forecast_svr_5aa39533ae1f941be7165ecd_MI_lag1.pkl
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
    five = 'lag' + str(lag) + '.pkl'
    
    name = one + two + three + four + five 
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
    print('#############')
    if impute:
        print('imputed')
    else:
        print('not imputed')
    print('Lag: {0}'.format(lag))
    print('#############')

    # Directory path
    dirpath = dirLagPath(lag=lag)

    product = genData(mongoid, conn, c, impute=impute, freq='daily')
    X_train, y_train, X_test, y_test, dftrain, scaler = splitDataNN(product, percent=0.2, n_in=lag, n_out=lag)

    regressor = SVR(kernel='poly')
    # Grid Search
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C':[1, 5, 10, 100], 'kernel':['linear']},
            {'C':[1, 5, 10, 100], 'kernel':['rbf'], 'gamma':[0.5, 0.1, 0.01, 0.001, 0.0001]}
            ]
    grid_search = GridSearchCV(estimator=regressor,
            param_grid=parameters,
            scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
            cv=5,
            n_jobs=-1,
            refit='r2',
            verbose=0)
    
    print('Grid Search Cross-Validation')
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    # k-fold cross validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=best_estimator, X=X_train, y=y_train, cv=10, n_jobs=-1,
            scoring='neg_mean_squared_error', verbose=0)
    accuracies.mean(), accuracies.std()

    svrname = makeModelName(mongoid, algo='svr', impute=impute, multivariate=True, lag=lag)
    print('{0}'.format(svrname))

    if save:
        joblib.dump(best_estimator, dirpath + svrname)
    else:
        pass

    evalForecast(best_estimator, X_train, y_train, inverse=True, scaler=scaler)

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
#for i in listOfIds:
#    makeModel(i, impute=True, inverse=True, save=True, lag=5)

# Multiviarate-Not Imputed
for i in listOfIds:
    makeModel(i, impute=False, inverse=True, save=True, lag=5)
