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

# MULTI-LAYER PERCEPTRON
#####################
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [
        {'vect__ngram_range': [(1,1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [str.split],
            'clf__hidden_layer_sizes': [(64,), (70,), (90,)],
            'clf__activation': ['logistic'],
            'clf__solver': ['adam'],
            'clf__batch_size': ['auto'],
            'clf__early_stopping': [True]
            }, 
        {'vect__ngram_range': [(1,1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [str.split],
            'vect__use_idf':[False],
            'vect__norm': [None],
            'clf__hidden_layer_sizes': [(64,), (70,), (90,)],
            'clf__activation': ['logistic'],
            'clf__solver': ['adam'],
            'clf__batch_size': ['auto'],
            'clf__early_stopping': [True]
            }
    ]

lr_tfidf = Pipeline(
        [   ('vect', tfidf),
            ('clf', MLPClassifier(random_state=42))
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
{'clf__activation': 'logistic',
 'clf__batch_size': 'auto',
 'clf__early_stopping': True,
 'clf__hidden_layer_sizes': 90,
 'clf__solver': 'adam',
 'vect__ngram_range': (1, 1),
 'vect__norm': None,
 'vect__stop_words': None,
 'vect__tokenizer': <method 'split' of 'str' objects>,
 'vect__use_idf': False}
-----------------------------------------------------
{'clf__activation': 'logistic',
 'clf__batch_size': 'auto',
 'clf__early_stopping': True,
 'clf__hidden_layer_sizes': (70,),
 'clf__solver': 'adam',
 'vect__ngram_range': (1, 1),
 'vect__norm': None,
 'vect__stop_words': None,
 'vect__tokenizer': <method 'split' of 'str' objects>,
 'vect__use_idf': False}

gs_lr_tfidf.best_score_
0.9925088590973765
-----------------------
0.9925399254820985
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
0.9922542430800775
0.9930825946483307
0.9926994449507083
0.9923161361141603
0.9918294223639791
------------------
0.9921817560500782
0.9927098003479413
0.9921506088973573
0.9925646707951039
0.9921711575502765


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
array([[  9189,    751,    113],
       [   469, 106544,   1220],
       [    98,    966, 363487]])
--------------------------------
array([[  9201,    729,    123],
       [   489, 106464,   1280],
       [    94,    887, 363570]])
'''
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# precision/recall
from sklearn.metrics import precision_score, recall_score, f1_score

(precision_score(y_train, y_train_pred, average='micro'),
        recall_score(y_train, y_train_pred, average='micro'))
'''
(0.9925088590973765, 0.9925088590973765)
----------------------------------------
(0.9925399254820985, 0.9925399254820985)
'''

(precision_score(y_train, y_train_pred, average='macro'),
        recall_score(y_train, y_train_pred, average='macro'))
'''
(0.9741227472911742, 0.9651772083028692)
----------------------------------------
(0.973872303861589, 0.9654046098906717)
'''

(f1_score(y_train, y_train_pred, average='micro'), 
        f1_score(y_train, y_train_pred, average='macro'))
'''
(0.9925088590973765, 0.9695803926915509)
----------------------------------------
(0.9925399254820985, 0.9695813093248251)
'''

# ROC curve --> multiclass format is not supported

# TEST MODEL ON TEST DATA
#########################
y_test_pred = cross_val_predict(best_estimator, X_test, y_test, cv=5,
        verbose=2, n_jobs=-1)

conf_mx = confusion_matrix(y_test, y_test_pred)
'''
array([[ 2008,   422,    83],
       [  262, 26138,   659],
       [   75,   428, 90635]])
----------------------------------------
array([[ 2005,   417,    91],
       [  287, 26111,   661],
       [   81,   418, 90639]])
'''

(precision_score(y_test, y_test_pred, average='micro'),
        recall_score(y_test, y_test_pred, average='micro'))
'''
(0.984019550989976, 0.984019550989976)
----------------------------------------
(0.9838041587275288, 0.9838041587275288)
'''

(precision_score(y_test, y_test_pred, average='macro'),
        recall_score(y_test, y_test_pred, average='macro'))
'''
(0.938891431300792, 0.919829709577983)
----------------------------------------
(0.9352352521774864, 0.9191138021202141)
'''

(f1_score(y_test, y_test_pred, average='micro'), 
        f1_score(y_test, y_test_pred, average='macro'))
'''
(0.984019550989976, 0.929029503925976)
----------------------------------------
(0.9838041587275288, 0.9269476906934734)
'''

# save fitted model to file
#joblib.dump(best_estimator, './training/sentiment_mlp2.pkl')
joblib.dump(best_estimator, './training/sentiment_mlp.pkl')
