'''
validasi model salescluster/group classification
'./training/grupClass_logreg.pkl'
'./training/grupClass_destree.pkl'
'./training/grupClass_svc.pkl'
'''

# library
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sqlite3

def gcModVal(X, y, folder='./training/', model='best'):
    '''
    group classification model validation
    '''
    listOfModels = {
            'best': folder + 'grupClass_svc.pkl',
            'destree': folder + 'grupClass_destree.pkl',
            'logreg': folder + 'grupClass_logreg.pkl'
            }

    if model == 'best':
        model = joblib.load(listOfModels['best'])
        modelname = 'C-Support Vector Classification Model'
    elif model == 'destree':
        model = joblib.load(listOfModels['destree'])
        modelname = 'Decision Tree Model'
    elif model == 'logreg':
        model = joblib.load(listOfModels['logreg'])
        modelname = 'Logistic Regression Model'
    else:
        print('Model you input is not supported!')

    # cross validation
    cvscore = cross_val_score(model, X, y, cv=5, verbose=0, n_jobs=-1)
    # prediction
    y_val_pred = cross_val_predict(model, X, y, cv=5, verbose=0, n_jobs=-1)
    # confusion matrix
    conf_mx = confusion_matrix(y, y_val_pred)

    # printing
    print('\n')
    print('{0}'.format(modelname))
    print('Cross Validation: {0}'.format(cvscore))
    print('Confusion Matrix: \n{0}'.format(conf_mx))
    print('\n')
    # Precision-recall with average='micro'
    print('Precision-Recall (\'micro\')')
    print('Precision: {0} \nRecall: {1}'.format(
        precision_score(y, y_val_pred, average='micro'),
        recall_score(y, y_val_pred, average='micro')))
    # f1 score
    print('F1 Score')
    print('F1 score (\'micro\'): {0}'.format(
        f1_score(y, y_val_pred, average='micro')))

# make connection to sqlite db
conn = sqlite3.connect('validasi.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# result reproducibility
np.random.seed(42)

# query
# get ranking, reviewcount, salescluster from table 'prodpage'
sqlc = 'SELECT ranking, reviewcount, salescluster FROM prodpage'
c.execute(sqlc)
conn.commit()
product = c.fetchall()
product = pd.DataFrame(product)
product.columns = ['ranking', 'reviewcount', 'salescluster']

X = product[['ranking', 'reviewcount']].values
y = product['salescluster'].values

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

allModels = ['best', 'destree', 'logreg']

for i in allModels:
    gcModVal(X, y, folder='./training/', model=i)
