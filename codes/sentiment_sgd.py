'''
'''
# library
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


# The text data is already cleaned
inputfile = './csvfiles/output_sentiment_outofcore.csv'
# The text data is already cleaned
inputfile = './csvfiles/output_sentiment.csv'
review = pd.read_csv(inputfile, skip_blank_lines=False)
review = review[['text', 'ovsentiment']]
# exclude NaN in 'text' column (count: 11248)
review = review[~pd.isna(review['text'])]
X = review['text'].values
y = review['ovsentiment'].values

# len(review[review['ovsentiment'] == -1])
# total number of -1 (negative review) is 12566

# len(review[review['ovsentiment'] == 0])
# total number of 0 (neutral) is 135292

# len(review[review['ovsentiment'] == 1])
# total number of 1 (positive review) is 455689

# splitting the dataset into the training set and test set
# stratify=y --> stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
        random_state=42, shuffle=True, stratify=y)

# STOCHASTIC GRADIENT DESCENT
#####################
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [
        {'vect__ngram_range': [(1,1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [str.split],
            'clf__loss': ['modified_huber', 'squared_hinge', 'perceptron', 'log'],
            'clf__penalty': ['l2', 'l1', 'elasticnet'],
            'clf__max_iter': [200, 300, 400, 500, 600]
            }, 
        {'vect__ngram_range': [(1,1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [str.split],
            'vect__use_idf':[False],
            'vect__norm': [None],
            'clf__loss': ['modified_huber', 'squared_hinge', 'perceptron', 'log'],
            'clf__penalty': ['l2', 'l1', 'elasticnet'],
            'clf__max_iter': [200, 300, 400, 500, 600]
            }
    ]

lr_tfidf = Pipeline(
        [   ('vect', tfidf),
            ('clf', SGDClassifier(random_state=42))
            ]
        )

gs_lr_tfidf = GridSearchCV(estimator=lr_tfidf,
        param_grid=param_grid,
        scoring=['accuracy', 'f1_macro', 'f1_micro'],
        cv=5,
        verbose=2,
        refit='f1_micro',
        n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

best_parameters = gs_lr_tfidf.best_params_
best_estimator = gs_lr_tfidf.best_estimator_
result = gs_lr_tfidf.cv_results_

'''
best_estimator.classes_
    array([-1,  0,  1])

best_parameters
{'clf__loss': 'modified_huber',
 'clf__max_iter': 500,
 'clf__penalty': 'l1',
 'vect__ngram_range': (1, 1),
 'vect__norm': None,
 'vect__stop_words': None,
 'vect__tokenizer': <method 'split' of 'str' objects>,
 'vect__use_idf': False}

best_estimator


gs_lr_tfidf.best_score_
    0.9836321574361534
'''

# PERFORMANCE MEASURE
#####################
# Stratified k-fold CV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train):
    clone_estimator = clone(best_estimator)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train[test_index])

    clone_estimator.fit(X_train_folds, y_train_folds)
    y_pred = clone_estimator.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

'''
0.984508486160155
0.984674012095104
0.9834210090299064
0.9839591574674316
0.9846012053932026

Is it good??
Let's try dumb classifier
'''
from sklearn.base import BaseEstimator
class DumbClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
dumb = DumbClassifier()
cross_val_score(dumb, X_train, y_train, cv=5, scoring='accuracy')
'''
array([0.22451537, 0.22286886, 0.22508725, 0.22405169, 0.22427952])
'''

# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(best_estimator, X_train, y_train, cv=5,
        verbose=2, n_jobs=-1)

conf_mx = confusion_matrix(y_train, y_train_pred)
'''
array([[  7732,   2095,    226],
       [  1458, 104573,   2202],
       [   218,   1704, 362629]])
'''

# precision/recall
from sklearn.metrics import precision_score, recall_score, f1_score

(precision_score(y_train, y_train_pred, average='micro'),
        recall_score(y_train, y_train_pred, average='micro'))
'''
(0.9836321574361534, 0.9836321574361534)
'''

(precision_score(y_train, y_train_pred, average='macro'),
        recall_score(y_train, y_train_pred, average='macro'))
'''
(0.9267158483496795, 0.910011823846748)
'''

(f1_score(y_train, y_train_pred, average='micro'), 
        f1_score(y_train, y_train_pred, average='macro'))
'''
(0.9836321574361534, 0.9180722700799406)
'''

# ROC curve --> multiclass format is not supported

# TEST MODEL ON TEST DATA
#########################
y_test_pred = cross_val_predict(best_estimator, X_test, y_test, cv=5,
        verbose=2, n_jobs=-1)

conf_mx = confusion_matrix(y_test, y_test_pred)
'''
array([[ 1935,   526,    52],
       [  329, 26100,   630],
       [   80,   606, 90452]])
'''

(precision_score(y_test, y_test_pred, average='micro'),
        recall_score(y_test, y_test_pred, average='micro'))
'''
(0.9815839615607654, 0.9815839615607654)
'''

(precision_score(y_test, y_test_pred, average='macro'),
        recall_score(y_test, y_test_pred, average='macro'))
'''
(0.9254865722936, 0.9090093001953637)
'''

(f1_score(y_test, y_test_pred, average='micro'), 
        f1_score(y_test, y_test_pred, average='macro'))
'''
(0.9815839615607654, 0.9169227343446013)
'''

# save fitted model to file
from sklearn.externals import joblib
joblib.dump(best_estimator, './training/sentiment_sgd.pkl')

