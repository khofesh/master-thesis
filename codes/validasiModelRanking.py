'''
A script to validate ranking model (random forest)
* y: sales rank = ranking
* x: discount value = cashback
     discount rate = cashbackval
     current price = price
     review valence = prodrating
     review volume = reviewcount
     % of negative review = negreview
     % of positive review = posreview
     number of answered questions = answerscnt
     number of people who find reviews helpful = otheragreemean
     rating of the most helpful positive review = ratingmosthelpful
     positive sentiment strength = possentiment
     negative sentiment strength = negsentiment
     sentiment polarity = sentipolarity
     reviewers' reputation / smiley = reviewersrep
     picture of reviewer = revpictotal
     picture of products = prodpicstotal
     goldbadge = merchanttype
     topads = topads
'''

import pandas as pd
import numpy as np
import sqlite3
import matplotlib as mpl
import matplotlib.pyplot as plt
# import validasiVariable.py
import validasiVariable as valVar
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

#Adj R square
def adj_r2_score(model,X_test, y_test,):
    y_pred = model.predict(X_test)
    # model.coefs_ doesn't exist
    adj = 1 - float(len(y_test)-1)/(len(y_test)-model.n_features_-1) * \
            (1 - r2_score(y_test,y_pred))
    return adj

# Evalute random search
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_adjusted = adj_r2_score(model, X_test, y_test)
    print('Model Validation')
    print('MSE: {0}'.format(MSE))
    print('R^2: {0}'.format(r2))
    print('R^2 Adjusted: {0}'.format(r2_adjusted))

# make connection to sqlite db
conn = sqlite3.connect('validasi.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# rank product
csvoutput = './csvfiles/validasi_ranking.csv'
valVar.meanRank(conn, c, csvoutput)

# get training data from database
# the output is dataframe
dftrain = valVar.prodpageTrain(conn, c)

# preprocessing
dftrain['merchanttype'] = dftrain['merchanttype'].astype('category')
dftrain['merchantname'] = dftrain['merchantname'].astype('category')
dftrain['topads'] = dftrain['topads'].replace((1,0), ('yes', 'no'))
dftrain['topads'] = dftrain['topads'].astype('category')
# drop column 'id' and 'prodname'
dftrain = dftrain.drop(['id', 'prodname', 'merchantname', 'actualrevcount'], axis=1)

# select independent and dependent variables
X = dftrain.iloc[:, :-1].values
y = dftrain.iloc[:, 18].values

# Encoding categorical data ('merchanttype' and 'topads')
labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
# 'merchanttype'
X[:, 0] = labelencoder1.fit_transform(X[:, 0])
# 'topads'
X[:, 1] = labelencoder2.fit_transform(X[:, 1])
# onehotencoder for both 'merchanttype'
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# dummy variables for merchanttype (3-1)
# avoiding dummy variable trap
X = X[:, 1:]

# feature scaling
sc1 = StandardScaler()
sc2 = StandardScaler()
X = sc1.fit_transform(X)
y = sc2.fit_transform(y.reshape(-1, 1))
y = y.ravel()

# load model
model = joblib.load('./training/regressor_randforest.pkl')

# validate chosen model
evaluate(model, X, y)
