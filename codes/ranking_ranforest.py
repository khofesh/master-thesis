'''
A script to train machine learning models.
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

# read csv file
dataset = pd.read_csv('./csvfiles/output_training.csv')
dataset['merchanttype'] = dataset['merchanttype'].astype('category')
dataset['merchantname'] = dataset['merchantname'].astype('category')
dataset['topads'] = dataset['topads'].replace((1,0), ('yes', 'no'))
dataset['topads'] = dataset['topads'].astype('category')
# drop column 'id' and 'prodname'
dataset = dataset.drop(['id', 'prodname', 'merchantname', 'actualrevcount'], axis=1)
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
y = dataset.iloc[:, 18].values

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

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# RANDOM FOREST
###############
regressor = RandomForestRegressor(random_state=42)

# Evalute random search
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Model Performance')
    print('MSE: {0}'.format(MSE))
    print('R^2: {0}'.format(r2))

#Adj R square
def adj_r2_score(model,X_test, y_test,):
    y_pred = model.predict(X_test)
    adj = 1 - float(len(y_test)-1)/(len(y_test)-len(model.coefs_)-1) * \
            (1 - r2_score(y_test,y_pred))
    return adj

# GRID SEARCH
parameters = [{
    'bootstrap': [True],
    'max_depth': [5, 7, 9, 10],
    'max_features': ['auto', 'log2'],
    'min_samples_leaf': [4, 5, 6],
    'min_samples_split': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'n_estimators': [100, 200, 300, 350, 400, 450, 500, 1000]
    } 
    ]

grid_search = GridSearchCV(estimator=regressor,
        param_grid=parameters,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=10,
        n_jobs=-1,
        refit='neg_mean_squared_error',
        verbose=2)

# model fitting using grid search parameter
grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_estimator = grid_search.best_estimator_
result = grid_search.cv_results_

# change random_state from None to 42
best_estimator = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
        max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=6, min_samples_split=14,
        min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
        oob_score=False, random_state=42, verbose=0, warm_start=False)

best_estimator.fit(X_train, y_train)
'''
best_parameters:
    {'bootstrap': True,
    'max_depth': 10,
    'max_features': 'auto',
    'min_samples_leaf': 6,
    'min_samples_split': 14,
    'n_estimators': 400}

best_estimator:
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=5, min_samples_split=9,
           min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

evaluate(best_estimator, X_train, y_train)
Model Performance
MSE: 0.2027189781371069
R^2: 0.7974347686890207

evaluate(best_estimator, X_test, y_test)
Model Performance
MSE: 0.23921619685787895
R^2: 0.7600476919853139
'''

# cross validation
accuracies = cross_val_score(estimator=best_estimator, X=X_train, y=y_train, cv=10, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=2)

accuracies.mean()
# -0.24467622291506017
accuracies.std()
# 0.00526904620084374

# save fitted model to file
joblib.dump(best_estimator, './training/regressor_randforest.pkl')

# FEATURE IMPORTANCES
#####################
importances = best_estimator.feature_importances_
'''
array([1.84112211e-03, 0.00000000e+00, 2.69915247e-04, 7.22695395e-05,
       1.62266747e-04, 2.82048738e-01, 6.79913509e-03, 6.72165968e-01,
       3.20918354e-04, 3.05775377e-04, 2.12051673e-02, 1.10947760e-06,
       1.42702209e-03, 2.21940418e-03, 2.32018880e-04, 0.00000000e+00,
       0.00000000e+00, 8.65384743e-03, 2.27532190e-03])
'''
std = np.std([tree.feature_importances_ for tree in best_estimator.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

'''
Feature ranking:
1. feature 7 (0.672166)
2. feature 5 (0.282049)
3. feature 10 (0.021205)
4. feature 17 (0.008654)
5. feature 6 (0.006799)
6. feature 18 (0.002275)
7. feature 13 (0.002219)
8. feature 0 (0.001841)
9. feature 12 (0.001427)
10. feature 8 (0.000321)
11. feature 9 (0.000306)
12. feature 2 (0.000270)
13. feature 14 (0.000232)
14. feature 4 (0.000162)
15. feature 3 (0.000072)
16. feature 11 (0.000001)
17. feature 15 (0.000000)
18. feature 16 (0.000000)
19. feature 1 (0.000000)
'''
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# RANDOMIZED SEARCH (NOT USED ANYMORE)
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
