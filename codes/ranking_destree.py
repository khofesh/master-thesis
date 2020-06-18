import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns

# read csv file
dataset = pd.read_csv('./csvfiles/output_training.csv')
dataset['merchanttype'] = dataset['merchanttype'].astype('category')
dataset['merchantname'] = dataset['merchantname'].astype('category')
dataset['topads'] = dataset['topads'].replace((1,0), ('yes', 'no'))
dataset['topads'] = dataset['topads'].astype('category')
# drop column 'id' and 'prodname'
dataset = dataset.drop(['id', 'prodname', 'merchantname'], axis=1)
# some 'revpictotal' has 9999 value and most of them have
# 'actualrevcount' = 10, only 5a986371b8a9f712ce73da7f has 119 'actualrevcount'
# replace 9999 with np.nan
dataset['revpictotal'] = dataset['revpictotal'].replace(9999, np.nan)
# data imputation
# mean technique
meanway = dataset['revpictotal'].mean()
dataset['revpictotal'] = dataset['revpictotal'].fillna(meanway)

# select independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 19].values

# Encoding categorical data ('merchanttype' and 'topads')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
sc3 = StandardScaler()
sc4 = StandardScaler()
X_train = sc1.fit_transform(X_train)
y_train = sc2.fit_transform(y_train.reshape(-1, 1))
y_train = y_train.ravel()
X_test = sc3.fit_transform(X_test)
y_test = sc4.fit_transform(y_test.reshape(-1, 1))
y_test = y_test.ravel()

# Decision trees
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(min_samples_split=5, random_state=42, max_depth=5)
regressor.fit(X_train, y_train)

# prediction
from sklearn.metrics import mean_squared_error, r2_score
y_pred = regressor.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
# --> 0.37737

# cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=2)
accuracies.mean()
accuracies.std()

# fit decision tree again
regressor.fit(X_train, y_train)

# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'min_samples_split': range(2, 15)} 
        ]
grid_search = GridSearchCV(estimator=regressor,
        param_grid=parameters,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=10,
        n_jobs=-1,
        refit='r2',
        verbose=2)

# model fitting using grid search parameter
grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_estimator = grid_search.best_estimator_
result = grid_search.cv_results_

# model fitting using best parameter
regressor = DecisionTreeRegressor(criterion='mse', max_depth=5, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=42, splitter='best')

regressor.fit(X_train, y_train)

# save fitted model to file
from sklearn.externals import joblib
joblib.dump(regressor, './training/regressor_destree.pkl')

# prediction
from sklearn.metrics import mean_squared_error, r2_score
y_pred = regressor.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
# --> 0.37737
r2_score(y_test, y_pred)
# --> 0.62263

############################################################
## Decision tree regression with AdaBoost
from sklearn.ensemble import AdaBoostRegressor
regressor2 = AdaBoostRegressor(DecisionTreeRegressor(criterion='mse', 
        max_depth=5, max_features=None,
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        presort=False, splitter='best'),
        n_estimators=500, random_state=42)

regressor2.fit(X_train, y_train)

# predict
y_pred = regressor2.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
# --> 0.37569
r2_score(y_test, y_pred)
# --> 0.6243



