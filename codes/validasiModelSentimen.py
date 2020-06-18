'''
Validasi model analisis sentimen
'''

import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def sentimentModelVal(csvfile, conn, cursor, startdate, model):
    '''
    csvfile: string. 
        path to csv file that contains preprocessed words and overall sentiment
    conn: sqlite3 connection
    cursor: sqlite3 cursor
    startdate: string. Format -> '2018-03-25'
        cut the dataset from the startdate to the most recent date
    model: string. 
        select model available.
        logreg - logistic regression
        mlp    - multilayer perceptron
        mlp2   - multilayer perceptron 
        nb     - naive bayes
        pa     - passive agressive
        sgd    - stochastic gradient descent
    '''
    # query
    sqlc = 'SELECT rowid, datereview FROM reviews'
    cursor.execute(sqlc)
    conn.commit()
    # get all the query
    datereview = cursor.fetchall()
    # convert it into dataframe
    datereview = pd.DataFrame(datereview)
    datereview.columns = ['rowid', 'date']
    # change 'date' string into datetime
    datereview['date'] = pd.to_datetime(datereview['date'], format='%d %b %Y')
    # raw format: 26 Jun 2018
    # %d -> Day of the month as a zero-padded decimal number. e.g.: 30
    # %b -> Month as locale’s abbreviated name. e.g.: Sep
    # %Y -> Year with century as a decimal number. e.g.: 2013

    # read csv file
    text = pd.read_csv(csvfile, skip_blank_lines=False)
    # merge text and datereview
    text = pd.merge(text, datereview, on='rowid')
    
    # select record => march 25 2018
    text = text[text['date'] >= startdate]
    # drop NaN
    text = text[~pd.isna(text['text'])]
    
    # X_val and y_val
    X_val = text['text'].values
    y_val = text['ovsentiment'].values

    # Model Path
    if model == 'logreg':
        modelpath = './training/sentiment_logreg.pkl'
        modelname = 'Logistic Regression'
    elif model == 'mlp':
        modelpath = './training/sentiment_mlp.pkl'
        modelname = 'MultiLayer Perceptron'
    elif model == 'mlp2':
        modelpath = './training/sentiment_mlp2.pkl'
        modelname = 'MultiLayer Perceptron'
    elif model == 'nb':
        modelpath = './training/sentiment_naivebayes.pkl'
        modelname = 'Naive Bayes'
    elif model == 'pa':
        modelpath = './training/sentiment_passiveagressive.pkl'
        modelname = 'Passive Agressive'
    elif model == 'sgd':
        modelpath = './training/sentiment_sgd.pkl'
        modelname = 'Stochastic Gradient Descent'
    else:
        print('Not Supported')
        return 0
    
    # load estimator
    estimator = joblib.load(modelpath)
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    # Cross validation
    cvscore = cross_val_score(estimator, X_val, y_val, cv=5, verbose=2, n_jobs=-1)
    
    # confusion matrix
    y_val_pred = cross_val_predict(estimator, X_val, y_val, cv=5, 
            verbose=2, n_jobs=-1)
    conf_mx = confusion_matrix(y_val, y_val_pred)

    # printing
    print('\n')
    print('{0}'.format(modelname))
    print('Cross Validation: {0}'.format(cvscore))
    print('Confusion Matrix: \n{0}'.format(conf_mx))
    print('\n')
    # Precision-recall with average='micro'
    print('Precision-Recall (\'micro\')')
    print('Precision: {0} \nRecall: {1}'.format(
        precision_score(y_val, y_val_pred, average='micro'),
        recall_score(y_val, y_val_pred, average='micro')))
    # Precision-recall with average='macro'
    print('Precision-Recall (\'macro\')')
    print('Precision: {0} \nRecall: {1}'.format(
        precision_score(y_val, y_val_pred, average='macro'),
        recall_score(y_val, y_val_pred, average='macro')))
    # f1 score
    print('F1 Score')
    print('F1 score (\'micro\'): {0} \nF1 score (\'macro\'): {1}'.format(
        f1_score(y_val, y_val_pred, average='micro'),
        f1_score(y_val, y_val_pred, average='macro')))

# make connection to sqlite db
conn = sqlite3.connect('validasi.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

csvfile = './csvfiles/output1_sentiment_validasi.csv'
startdate = '2018-03-25'
sentimentModelVal(csvfile, conn, c, startdate, model='nb')
