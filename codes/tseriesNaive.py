import tseriesRoutines as routines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import ceil
from math import sqrt
import sqlite3
# rpy2
# courtesy:
# https://stackoverflow.com/questions/36105701/how-to-run-auto-arima-using-rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
pandas2ri.activate()


# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

test = routines.sqlToDf(conn, c, 20)
test.prod.head()
product = test.selectReview2(test.prod.loc[3, 'id'])
product2 = routines.tsSalesRateSentiment(product, freq='daily', standardize=True)
product2 = product2[['Sales']]

# naive method
ro.r('library(forecast)')
rdf = pandas2ri.py2ri(product2)
ro.globalenv['r_timeseries'] = rdf
ored = ro.r('as.data.frame(forecast(naive(r_timeseries),h=10))')
