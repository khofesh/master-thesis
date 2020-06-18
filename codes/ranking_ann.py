'''
A script to train machine learning models.
* y: sales rank = salescluster
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
#import seaborn as sns

# read csv file
dataset = pd.read_csv('./csvfiles/output_training.csv')
dataset['merchanttype'] = dataset['merchanttype'].astype('category')
dataset['merchantname'] = dataset['merchantname'].astype('category')
dataset['topads'] = dataset['topads'].replace((1,0), ('yes', 'no'))
dataset['topads'] = dataset['topads'].astype('category')
# salescluster should start from 0-9
dataset['salescluster'] = dataset['salescluster'] - 1
dataset['salescluster'] = dataset['salescluster'].astype('int')
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

'''
Plotting:

sns.stripplot(x='merchanttype', y='ranking', data=dataset)
sns.stripplot(x='merchanttype', y='ranking', data=dataset, jitter=True)
sns.swarmplot(y='merchanttype', x='ranking', data=dataset)
sns.boxplot(x='merchanttype', y='ranking', data=dataset)
sns.boxplot(x='merchanttype', y='ranking', hue='topads', data=dataset)
sns.barplot(x='merchanttype', y='ranking', hue='topads', data=dataset)
sns.countplot(x='merchanttype', data=dataset, palette="Greens_d")
sns.countplot(y='merchanttype', data=dataset, hue='topads',  palette="Greens_d")
'''

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
# salescluster
#onehotencoder2 = OneHotEncoder(categorical_features=[0])
#y = onehotencoder2.fit_transform(y.reshape(-1, 1)).toarray()
import keras
y = keras.utils.to_categorical(y, num_classes=10)

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#### Training Phase

# Keras libraries and required packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# input node = 20, output node = 10 -> (20+10)/2
classifier.add(Dense(output_dim=15, kernel_initializer='uniform', activation='relu', input_dim=20))

# Applies Dropout to the input.
# Dropout consists in randomly setting a fraction rate of input 
# units to 0 at each update during training time, which helps prevent overfitting.
classifier.add(Dropout(0.5))

# Adding the second hidden layer
classifier.add(Dense(output_dim=15, kernel_initializer='uniform', activation='relu'))

# Dropout
classifier.add(Dropout(0.5))

# Adding the output layer
classifier.add(Dense(output_dim=10, kernel_initializer='uniform', activation='softmax'))

# Stochastic Gradient Descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# optimizer
classifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, epochs=100, batch_size=128)

score = classifier.evaluate(X_test, y_test, batch_size=128)

# save the model
classifier.save('./training/first.h5')

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred2 = np.argmax(y_pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred2)

#### Cross validation and Grid search

# Cross-validation

