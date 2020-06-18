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

# PASSIVE AGRESSIVE
#####################
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [
        {'vect__ngram_range': [(1,1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [str.split],
            'clf__loss': ['hinge', 'squared_hinge'],
            'clf__shuffle': [True, False]
            }, 
        {'vect__ngram_range': [(1,1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [str.split],
            'vect__use_idf':[False],
            'vect__norm': [None],
            'clf__loss': ['hinge', 'squared_hinge'],
            'clf__shuffle': [True, False]
            }
    ]

lr_tfidf = Pipeline(
        [   ('vect', tfidf),
            ('clf', PassiveAggressiveClassifier(random_state=42))
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
{'clf__loss': 'hinge',
 'clf__shuffle': True,
 'vect__ngram_range': (1, 1),
 'vect__stop_words': None,
 'vect__tokenizer': <method 'split' of 'str' objects>}
-----------------------------------------------------

gs_lr_tfidf.best_score_
0.98115720211997
-----------------------
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
------------------


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
array([[  7778,   1990,    285],
       [  1501, 103101,   3631],
       [   230,   1461, 362860]])
--------------------------------
'''

# precision/recall
from sklearn.metrics import precision_score, recall_score, f1_score

(precision_score(y_train, y_train_pred, average='micro'),
        recall_score(y_train, y_train_pred, average='micro'))
'''
(0.98115720211997, 0.98115720211997)
----------------------------------------
'''

(precision_score(y_train, y_train_pred, average='macro'),
        recall_score(y_train, y_train_pred, average='macro'))
'''
(0.9249657241324081, 0.9072148626674309)
----------------------------------------
'''

(f1_score(y_train, y_train_pred, average='micro'), 
        f1_score(y_train, y_train_pred, average='macro'))
'''
(0.98115720211997, 0.9158624790178168)
----------------------------------------
'''

# ROC curve --> multiclass format is not supported

# TEST MODEL ON TEST DATA
#########################
y_test_pred = cross_val_predict(best_estimator, X_test, y_test, cv=5,
        verbose=2, n_jobs=-1)

conf_mx = confusion_matrix(y_test, y_test_pred)
'''
array([[ 1670,   742,   101],
       [  384, 25646,  1029],
       [   84,   494, 90560]])
----------------------------------------
'''

(precision_score(y_test, y_test_pred, average='micro'),
        recall_score(y_test, y_test_pred, average='micro'))
'''
(0.9765222433932566, 0.9765222433932566)
----------------------------------------
'''

(precision_score(y_test, y_test_pred, average='macro'),
        recall_score(y_test, y_test_pred, average='macro'))
'''
(0.907600325954767, 0.8686610379519951)
----------------------------------------
'''

(f1_score(y_test, y_test_pred, average='micro'), 
        f1_score(y_test, y_test_pred, average='macro'))
'''
(0.9765222433932566, 0.8865579361878589)
----------------------------------------
'''

# save fitted model to file
from sklearn.externals import joblib
joblib.dump(best_estimator, './training/sentiment_passiveagressive.pkl')
