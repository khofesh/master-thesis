'''
Multilayer Perceptron Regressor
'''
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


# read csv file
dataset = pd.read_csv('./csvfiles/output_training.csv')
dataset['merchanttype'] = dataset['merchanttype'].astype('category')
dataset['merchantname'] = dataset['merchantname'].astype('category')
dataset['topads'] = dataset['topads'].replace((1,0), ('yes', 'no'))
dataset['topads'] = dataset['topads'].astype('category')
# drop column 'id', 'prodname', 'merchantname', and 'actualrevcount'
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

# Evalute model
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

# MULTI-LAYER PERCEPTRON
#####################
from sklearn.neural_network import MLPRegressor

regressor = MLPRegressor(random_state=42)

# GRID SEARCH
parameters = [{
    'hidden_layer_sizes': [(8, ), (32,), (64,), (120, 120)],
    'activation': ['logistic', 'relu'],
    'solver': ['adam', 'lbfgs'],
    'batch_size': ['auto', 8, 16, 32, 64, 128],
    'early_stopping': [True]
    } 
    ]

grid = GridSearchCV(estimator=regressor,
        param_grid=parameters,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=10,
        n_jobs=-1,
        refit='neg_mean_squared_error',
        verbose=2
    )

grid_result = grid.fit(X_train, y_train)

best_parameters = grid_result.best_params_
best_estimator = grid_result.best_estimator_
result = grid_result.cv_results_
'''
best_parameters
{'activation': 'relu',
 'batch_size': 'auto',
 'early_stopping': True,
 'hidden_layer_sizes': (90,),
 'solver': 'lbfgs'}
-----------------------------
{'activation': 'relu',
 'batch_size': 'auto',
 'early_stopping': True,
 'hidden_layer_sizes': (110,),
 'solver': 'lbfgs'}
-----------------------------
{'activation': 'relu',
 'batch_size': 'auto',
 'early_stopping': True,
 'hidden_layer_sizes': (120,),
 'solver': 'lbfgs'}
    
evaluate(best_estimator, X_train, y_train)
Model Performance
MSE: 0.25163314360154765
R^2: 0.7483668563984525
------------------------------------------
Model Performance
MSE: 0.2495933355208044
R^2: 0.7504066644791957
------------------------------------------
Model Performance
MSE: 0.24968594250355045
R^2: 0.7503140574964496

adj_r2_score(best_estimator, X_train, y_train)
0.7483517310329247
----------------------------------------------
0.7503916617240777
----------------------------------------------
0.7502990491748373

'''

# cross validation
accuracies = cross_val_score(estimator=best_estimator, X=X_train, y=y_train, cv=10, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=2)

accuracies.mean()
#-0.2577117063318107 
accuracies.std()
#0.004995898007277126 

'''
evaluate(best_estimator, X_test, y_test)
Model Performance
MSE: 0.346298128045242
R^2: 0.653701871954758

adj_r2_score(best_estimator, X_test, y_test)
0.6536185971854793
'''

# save fitted model to file
joblib.dump(best_estimator, './training/regressor_mlp.pkl')
