'''
Regression using neural network
'''
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


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

#### Training Phase

# Keras libraries and required packages
from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import Adam
from keras.optimizers import Nadam
from tensorflow import set_random_seed
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm

# for reproducibility
np.random.seed(42)
set_random_seed(42)

# optimizers
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

# 1. BATCH_SIZE AND EPOCHS
##########################
# Grid Search Hyperparameters
def grid_model(optim=adam):
    model = models.Sequential()
    model.add(layers.Dense(30, activation='relu',
        input_shape=(X_train.shape[1], )))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
    return model

model = KerasRegressor(build_fn=grid_model, verbose=0)

# grid search parameters
batch_size = [20, 25, 30, 35, 40]
epochs = [150, 200, 250]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model,
        param_grid=param_grid,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=5,
        n_jobs=-1,
        refit='neg_mean_squared_error',
        verbose=2)

grid_result = grid.fit(X_train, y_train)

best_parameters = grid_result.best_params_
result = grid_result.cv_results_
'''
best_parameters: {'batch_size': 25, 'epochs': 200}
result['mean_test_r2'][4] -> 0.73531931
result['mean_test_explained_variance'][4] -> 0.73616649
result['mean_test_neg_mean_squared_error'][4] -> -0.26464800294993623
'''

# 2. DROPOUT REGULARIZATION
##########################
# Grid Search Hyperparameters
def grid_model2(optim=adam, dropout_rate=0.0, weight_constraint=0):
    model = models.Sequential()
    model.add(layers.Dense(30, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint),
        input_shape=(X_train.shape[1], )))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(30, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
    return model

model = KerasRegressor(build_fn=grid_model2, verbose=0, epochs=200, batch_size=25)

# grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)

grid = GridSearchCV(estimator=model,
        param_grid=param_grid,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=5,
        n_jobs=-1,
        refit='neg_mean_squared_error',
        verbose=2)

grid_result = grid.fit(X_train, y_train)

best_parameters = grid_result.best_params_
result = grid_result.cv_results_
'''
best_parameters: {'dropout_rate': 0.1, 'weight_constraint': 4}
8
result['mean_test_r2'][8] -> 0.740648243077989 
result['mean_test_explained_variance'][8] -> 0.7409284598730808
result['mean_test_neg_mean_squared_error'][8] -> -0.2593527756370157
'''

# 3. NUMBER OF NEURON IN THE HIDDEN LAYERS
##########################
# Grid Search Hyperparameters
def grid_model3(optim=adam, dropout_rate=0.1, weight_constraint=4, neurons=20):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint),
        input_shape=(X_train.shape[1], )))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
    return model

model = KerasRegressor(build_fn=grid_model3, verbose=0, epochs=200, batch_size=25)

# grid search parameters
neurons = [20, 24, 26, 28, 30, 35, 40, 45, 50, 60, 64]

param_grid = dict(neurons=neurons)

grid = GridSearchCV(estimator=model,
        param_grid=param_grid,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=5,
        n_jobs=-1,
        refit='neg_mean_squared_error',
        verbose=2)

grid_result = grid.fit(X_train, y_train)

best_parameters = grid_result.best_params_
result = grid_result.cv_results_
'''
best_parameters: {'neurons': 30}
4
result['mean_test_r2'][4] -> 0.7419171759725895
result['mean_test_explained_variance'][4] -> 0.7420599353705832
result['mean_test_neg_mean_squared_error'][4] -> -0.2580659280597288
'''

# 4. NETWORK WEIGHT INITIALIZATION
##########################
# Grid Search Hyperparameters
def grid_model4(optim=adam, dropout_rate=0.1, weight_constraint=4, neurons=30, init_mode='uniform'):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint),
        kernel_initializer=init_mode,
        input_shape=(X_train.shape[1], )))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint),
        kernel_initializer=init_mode))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
    return model

model = KerasRegressor(build_fn=grid_model4, verbose=0, epochs=200, batch_size=25)

# grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict(init_mode=init_mode)

grid = GridSearchCV(estimator=model,
        param_grid=param_grid,
        scoring=['neg_mean_squared_error', 'r2', 'explained_variance'],
        cv=5,
        n_jobs=-1,
        refit='neg_mean_squared_error',
        verbose=2)

grid_result = grid.fit(X_train, y_train)

best_parameters = grid_result.best_params_
result = grid_result.cv_results_
'''
best_parameters: {'init_mode': 'glorot_normal'}
4
result['mean_test_r2'][4] -> 0.7409485827604594
result['mean_test_explained_variance'][4] -> 0.7411534878427668
result['mean_test_neg_mean_squared_error'][4] -> -0.25903741254211166
'''

# ADDING MORE LAYER
def grid_model5(optim=adam, dropout_rate=0.1, weight_constraint=4, neurons=20):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint),
        input_shape=(X_train.shape[1], )))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(neurons, activation='relu', 
        kernel_constraint=maxnorm(weight_constraint)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
    return model



# Training the final model
model = grid_model5(optim=adam, dropout_rate=0.1, weight_constraint=4, neurons=30)
model.fit(X_train, y_train, epochs=200, batch_size=30, verbose=0)
loss, test_mse_score, test_mae_score = model.evaluate(X_test, y_test)

# Save fitted model to a file

#NOT USED
#
## General Model
#def gen_model(capacity, optim='rmsprop'):
#    model = models.Sequential()
#    model.add(layers.Dense(capacity, activation='relu',
#        input_shape=(X_train.shape[1], )))
#    model.add(layers.Dense(capacity, activation='relu'))
#    model.add(layers.Dense(1))
#    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
#    return model
#
## General Model 2
#def gen_model2(capacity, lreg, optim='rmsprop'):
#    model = models.Sequential()
#    model.add(layers.Dense(capacity, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=lreg, l2=lreg),
#        input_shape=(X_train.shape[1], )))
#    model.add(layers.Dense(capacity, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=lreg, l2=lreg)))
#    model.add(layers.Dense(1))
#    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
#    return model
#
## General Model 3
#def gen_model3(capacity, drop, optim='rmsprop'):
#    model = models.Sequential()
#    model.add(layers.Dense(capacity, activation='relu',
#        input_shape=(X_train.shape[1], )))
#    model.add(layers.Dropout(drop))
#    model.add(layers.Dense(capacity, activation='relu'))
#    model.add(layers.Dropout(drop))
#    model.add(layers.Dense(1))
#    model.compile(optimizer=optim, loss='mse', metrics=['mae', 'mse'])
#    return model
#
## K-fold cross-validation
#def kfoldcv(k, num_epochs, batch_size, X_train, y_train, model):
#    k = k
#    num_val_samples = len(X_train) // k
#    num_epochs = num_epochs
#    
#    all_mae_histories = []
#    all_mse_histories = []
#    
#    for fold in range(k):
#        print('processing fold #', fold)
#        val_data = X_train[num_val_samples * fold : num_val_samples * (fold + 1)]
#        val_targets = y_train[num_val_samples * fold : num_val_samples * (fold + 1)]
#    
#        partial_train_data = np.concatenate(
#                [X_train[:num_val_samples * fold],
#                    X_train[num_val_samples * (fold + 1):]],
#                axis=0)
#        partial_train_targets = np.concatenate(
#                [y_train[:num_val_samples * fold],
#                    y_train[num_val_samples * (fold + 1):]],
#                axis=0)
#    
#        model = model
#        history = model.fit(partial_train_data, partial_train_targets,
#                validation_data=(val_data, val_targets),
#                epochs=num_epochs, batch_size=batch_size, verbose=0)
#        mae_history = history.history['val_mean_absolute_error']
#        mse_history = history.history['val_mean_squared_error']
#        all_mae_histories.append(mae_history)
#        all_mse_histories.append(mse_history)
#
#    return all_mae_histories, all_mse_histories, history
#
## smooth curve
#def smooth_curve(points, factor=0.9):
#    smoothed_points = []
#    for point in points:
#        if smoothed_points:
#            previous = smoothed_points[-1]
#            smoothed_points.append(previous * factor + point * (1-factor))
#        else:
#            smoothed_points.append(point)
#    return smoothed_points
#
#k = 6
#num_epochs = 200
#batch_size = 30
#
#all_mae_histories, all_mse_histories, model = kfoldcv(k, num_epochs, batch_size, X_train, y_train, gen_model())
#
#np.average(all_mae_histories)
#np.average(all_mse_histories)
#
#average_mae_history = [
#        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
#
#average_mse_history = [
#        np.mean([x[i] for x in all_mse_histories]) for i in range(num_epochs)]
#
## plotting smooth curve (MAE)
#smooth_mae_history = smooth_curve(average_mae_history[10:])
#plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
#plt.xlabel('Epochs')
#plt.ylabel('Validation MAE')
#plt.show()
#
## plotting smooth curve (MSE)
#smooth_mse_history = smooth_curve(average_mse_history)
#plt.plot(range(1, len(smooth_mse_history) + 1), smooth_mse_history)
#plt.xlabel('Epochs')
#plt.ylabel('Validation MSE')
#plt.show()
#
## plotting validation scores
#import matplotlib.pyplot as plt
#plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
#plt.xlabel('Epochs')
#plt.ylabel('Validation MAE')
#plt.show()
#
#
