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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


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

# feature scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
X = sc1.fit_transform(X)
y = sc2.fit_transform(y.reshape(-1, 1))
y = y.ravel()

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


# EXPLORATORY DATA ANALYSIS
###########################
X_df = pd.DataFrame(X)
X_df.columns = ['merchanttype1', 'merchanttype2', 'topads', 'cashback', 'cashbackval', 'price',
'prodrating', 'reviewcount', 'negreview', 'posreview', 'answerscnt',
'otheragreemean', 'ratingmosthelpful', 'possentiment', 'negsentiment',
'sentipolarity', 'reviewersrep', 'revpictotal', 'prodpicstotal']
X_df['ranking'] = pd.DataFrame(y)

# pair plot
cols = ['sentipolarity', 'reviewersrep', 'revpictotal', 'prodpicstotal', 'ranking']
sns.pairplot(X_df[cols], size=2.5)
plt.tight_layout()
plt.show()

# correlation coefficient
cm = np.corrcoef(X_df.values.T)
sns.set(font_scale=0.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, yticklabels=X_df.columns, xticklabels=X_df.columns)
plt.show()


# LINEAR REGRESSION
###########################
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X_train, y_train)

# robust regression using RANSAC
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
        max_trials=300,
        min_samples=50,
        loss='absolute_loss',
        residual_threshold=5.0,
        random_state=42)
ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# model evaluation
from sklearn.metrics import mean_squared_error, r2_score
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
y_train_pred_ransac = ransac.estimator_.predict(X_train)
y_test_pred_ransac = ransac.estimator_.predict(X_test)

def residuals_plot(y_train_pred, y_train, y_test_pred, y_test):
    plt.scatter(y_train_pred, y_train_pred - y_train, 
            c='steelblue', marker='o', edgecolors='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, 
            c='limegreen', marker='s', edgecolors='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.show()
    return

def evaluate(y_train_pred, y_train, y_test_pred, y_test):
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

'''
evaluate(y_train_pred, y_train, y_test_pred, y_test)
    MSE train: 0.731, test: 0.739
    R^2 train: 0.269, test: 0.258
evaluate(y_train_pred_ransac, y_train, y_test_pred_ransac, y_test)
    MSE train: 0.800, test: 0.829
    R^2 train: 0.200, test: 0.168
'''

# Using regularized methods for regression
# Ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
# LASSO
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
# Elastic Net
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet.fit(X_train, y_train)

'''
evaluate(ridge.predict(X_train), y_train, ridge.predict(X_test), y_test)
    MSE train: 0.731, test: 0.739
    R^2 train: 0.269, test: 0.259
evaluate(lasso.predict(X_train), y_train, lasso.predict(X_test), y_test)
    MSE train: 1.001, test: 0.997
    R^2 train: 0.000, test: -0.000
evaluate(elanet.predict(X_train), y_train, elanet.predict(X_test), y_test)
    MSE train: 1.001, test: 0.997
    R^2 train: 0.000, test: -0.000
'''

# POLYNOMIAL REGRESSION
###########################
from sklearn.preprocessing import PolynomialFeatures
regr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_train_quad = quadratic.fit_transform(X_train)
X_train_cubic = cubic.fit_transform(X_train)
X_test_quad = quadratic.fit_transform(X_test)
X_test_cubic = cubic.fit_transform(X_test)

regr = regr.fit(X_train_quad, y_train)
'''
r2_score(y_train, regr.predict(X_train_quad))
    0.43451638175972107
r2_score(y_test, regr.predict(X_test_quad))
    -3.63582760096175e+17
'''

regr = regr.fit(X_train_cubic, y_train)
'''
r2_score(y_train, regr.predict(X_train_cubic))
    0.5556589766006779
r2_score(y_test, regr.predict(X_test_cubic))
    -9516542157354032.0
'''


