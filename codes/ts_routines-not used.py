'''
Related to time series.
plotting, grouping, etc.
'''

# library
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import math
from scipy.stats import zscore
from statsmodels.tsa.stattools import grangercausalitytests
# modules for terasvirta test
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import FloatVector
# r package 'forecast'


# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

def selectReview(mongoid, connection, cursor, impute=False):
    '''
    Select rowid, id, datereview, timereview
        based on mongoid and return dataframe
    mongoid: must be string. mongodb id.
    '''
    mongoid = '"' + mongoid + '"'
    #sqlc = 'SELECT rowid, id, datereview, timereview from reviews WHERE id = '
    sqlc = 'SELECT rowid, id, datereview from reviews WHERE id = '
    sqlc = sqlc + mongoid
    # execute
    cursor.execute(sqlc)
    connection.commit()

    # get all the query result and turn it into dataframe
    product = cursor.fetchall()
    product = pd.DataFrame(product)
    #product.columns = ['rowid', 'id', 'date', 'timereview']
    product.columns = ['rowid', 'id', 'date']

    # combine datereview and timereview
    #product['date'] = product['datereview'] + ' ' + product['timereview']
    # select only 'rowid', 'id', and 'date'
    #product = product[['rowid', 'id', 'date']]
    # change string to date format
    #product['date'] = pd.to_datetime(product['date'], format='%Y-%m-%d %H:%M %Z')
    product['date'] = pd.to_datetime(product['date'], format='%Y-%m-%d')

    # select record before march 01 2018
    product = product[product['date'] < '2018-03-01']

    '''
    python strftime explanation:
    %Y = Year with century as a decimal number. ~ 2013
    %m = Month as a zero-padded decimal number. ~ 09
    %d = Day of the month as a zero-padded decimal number. ~ 15
    %H = Hour (24-hour clock) as a zero-padded decimal number. ~ 13
    %M = Minute as a zero-padded decimal number. ~ 05
    %Z = Time zone name (empty string if the object is naive). ~ WIB
    '''

    # Minimal Purchase Quantity, product sold, 
    # and actual review count
    sqlc1 = 'SELECT prodsold, minbeli, actualrevcount FROM prodpage WHERE id = ' + mongoid
    cursor.execute(sqlc1)
    connection.commit()

    minbeli = cursor.fetchall()
    minbeli = pd.DataFrame(minbeli)
    minbeli.columns = ['prodsold', 'minbeli', 'actualrevcount']

    # assumption 1: 1 reviewer -> 1 * minbeli
    product['minpurchase'] = minbeli['minbeli'].values[0]

    # assumption 2: minimum date -> start date
    '''
    Problem:
        product sold != number of reviews
    Solution:
        1. sumneeded = (product sold/minbeli) - number of reviews
        2. date range = maximum date - minimum date 
        3. generate random multinomial number with number of
            experiment = date range and sum to sumneeded.
        4. generated number * minbeli
    Example:
        product sold = 134700
        number of reviews = 46
        sumneeded = 134700 - 134654
        date range = 1076
        test = np.random.multinomial(sumneeded, [1/daterange.]*daterange, size=1).flatten()
        ---> flatten() is needed, because Series requires 1-dimensional
        test = pd.Series(test)
    '''
    if impute and minbeli['prodsold'].values[0] > int(minbeli['actualrevcount'].values[0]):
        daterange = product['date'].max() - product['date'].min()
        # set minimum date as the start date
        startdate = str(product['date'].min().date())
        daterange = daterange.days
        actualrevcount = int(minbeli['actualrevcount'].values[0])
        # added
        sumneeded = math.ceil(minbeli['prodsold'].values[0]/minbeli['minbeli'].values[0]) - actualrevcount
        # generate imputed sales data
        imputed = np.random.multinomial(sumneeded, [1/float(daterange)]*daterange, size=1).flatten()
        imputed = pd.DataFrame(imputed)
        imputed.columns = ['imputedsales']
        imputed['imputedsales'] = imputed['imputedsales'] * minbeli['minbeli'].values[0]
        imputed['imputeddate'] = pd.Series(pd.date_range(start=startdate, periods=daterange))
        # merge product dataframe with imputed dataframe
        product = pd.merge(imputed, product[['id', 'date', 'minpurchase']], left_on='imputeddate', right_on='date', how='right')
        # set new index
        product.index = range(0, len(product))

        # check whether product['imputeddate'] is null or not,
        # if it is, replace it with product['date']
        for i in range(len(product)):
            # if it's null
            if pd.isnull(product.loc[i, 'imputeddate']):
                product.loc[i, 'imputeddate'] = product.loc[i, 'date']

        # replace NaN with 0
        product['imputedsales'] = product['imputedsales'].fillna(0)
        product['minpurchase'] = product['minpurchase'].fillna(0)

        # sum product['imputedsales'] and product['minpurchase']
        product['minpurchase'] = product['imputedsales'] + product['minpurchase']

        # we only need 'imputeddate', 'id', and 'minpurchase'
        product = product[['id', 'imputeddate', 'minpurchase']]
        # rename columns name
        product.columns = ['id', 'date', 'minpurchase']

        # convert sales to int
        product['Sales'] = product['Sales'].astype('int')

        # after all the processes above, return product
        return product

    else:
        pass

    product = product[['id', 'date', 'minpurchase']]
    product.columns = ['id', 'date', 'Sales']
    return product

def tsPlot(dataframe, columnname=None, freq='daily', result='plot', cumsum=False):
    '''
    dataframe columns name in order:
        'rowid' 'id' 'date' 'Sales'

    dataframe must contains the right 'date' format
    columnname = a string.
        date column's name
    freq = a string. supports only 'daily', 'weekly', and 'monthly'
    result = what you want this function to return.
        it could return a plot or a dataframe with date as index.
    '''
    dataframe.index = range(0, len(dataframe))
    mongoid = dataframe.loc[0, 'id']
    dataframe.columns = ['id', 'date', 'Sales']

    if freq == 'daily':
        product = dataframe.groupby(pd.Grouper(key='date', freq='D')).sum()
    elif freq == 'weekly':
        product = dataframe.groupby(pd.Grouper(key='date', freq='W')).sum()
    elif freq == 'monthly':
        product = dataframe.groupby(pd.Grouper(key='date', freq='M')).sum()
    elif cumsum & (freq == 'daily'):
        product = dataframe.groupby(pd.Grouper(key='date', freq='D')).cumsum()
    elif cumsum & (freq == 'weekly'):
        product = dataframe.groupby(pd.Grouper(key='date', freq='W')).cumsum()
    elif cumsum & (freq == 'monthly'):
        product = dataframe.groupby(pd.Grouper(key='date', freq='M')).cumsum()
    else:
        print('Not supported')
        print('The function supports only \'daily\', \'weekly\', and \'monthly\'')

    # groupby per day, then average
    #monthly = product['minpurchase'].groupby([lambda x: x.year, lambda x: x.month]).cumsum()

    if result == 'plot':
        return product.plot(title='id: ' + mongoid)
    elif result == 'df':
        return product
    else:
        print('Not supported')
        print('result parameter could either be \'plot\' or \'df\'')

def selectProd(connection, cursor, actualrevcount):
    '''
    Select product from prodpage table with actualrevcount => certain amount
        inputted in by user.
    returns dataframe.
    actualrevcount: integer.
    '''
    sqlc = 'SELECT id, prodsold, salescluster, actualrevcount FROM prodpage WHERE actualrevcount >= '
    sqlcall = sqlc + str(actualrevcount)
    cursor.execute(sqlcall)
    connection.commit()

    product = cursor.fetchall()
    product = pd.DataFrame(product)
    product.columns = ['id', 'prodsold', 'salescluster', 'actualrevcount']
    return product

def selectReview2(mongoid, connection, cursor, impute=False, rating=0, ovsentiment=0):
    '''
    selectReview with sentiment and rating.
    Select rowid, id, datereview, timereview
        based on mongoid and return dataframe
    mongoid: must be string. mongodb id.
    rating: if impute=True and there is no review, default rating is 0
    ovsentiment: if impute=True and buyer didn't leave review, 
        default overall sentiment is neutral/0
    '''
    mongoid = '"' + mongoid + '"'
    sqlc = 'SELECT rowid, id, datereview, rating, ovsentiment from reviews WHERE id = '
    sqlc = sqlc + mongoid
    # execute
    cursor.execute(sqlc)
    connection.commit()

    # get all the query result and turn it into dataframe
    product = cursor.fetchall()
    product = pd.DataFrame(product)
    product.columns = ['rowid', 'id', 'date', 'rating', 'ovsentiment']

    # detect date format
    product['date'] = pd.to_datetime(product['date'], format='%Y-%m-%d')

    # select record before march 01 2018
    product = product[product['date'] < '2018-03-01']

    # Minimal Purchase Quantity, product sold, 
    # and actual review count
    sqlc1 = 'SELECT prodsold, minbeli, actualrevcount FROM prodpage WHERE id = ' + mongoid
    cursor.execute(sqlc1)
    connection.commit()

    minbeli = cursor.fetchall()
    minbeli = pd.DataFrame(minbeli)
    minbeli.columns = ['prodsold', 'minbeli', 'actualrevcount']

    # assumption 1: 1 reviewer -> 1 * minbeli
    product['minpurchase'] = minbeli['minbeli'].values[0]

    if impute and minbeli['prodsold'].values[0] > int(minbeli['actualrevcount'].values[0]):
        daterange = product['date'].max() - product['date'].min()
        # set minimum date as the start date
        startdate = str(product['date'].min().date())
        daterange = daterange.days
        actualrevcount = int(minbeli['actualrevcount'].values[0])
        sumneeded = math.ceil(minbeli['prodsold'].values[0]/minbeli['minbeli'].values[0]) - actualrevcount
        # generate imputed sales data
        imputed = np.random.multinomial(sumneeded, [1/float(daterange)]*daterange, size=1).flatten()
        imputed = pd.DataFrame(imputed)
        imputed.columns = ['imputedsales']
        imputed['imputedsales'] = imputed['imputedsales'] * minbeli['minbeli'].values[0]
        imputed['imputeddate'] = pd.Series(pd.date_range(start=startdate, periods=daterange))
        # merge product dataframe with imputed dataframe
        product = pd.merge(imputed, product[['id', 'date', 'minpurchase', 'rating', 'ovsentiment']], left_on='imputeddate', right_on='date', how='right')
        # set new index
        product.index = range(0, len(product))

        # check whether product['imputeddate'] is null or not,
        # if it is, replace it with product['date']
        for i in range(len(product)):
            # if it's null
            if pd.isnull(product.loc[i, 'imputeddate']):
                product.loc[i, 'imputeddate'] = product.loc[i, 'date']

        # replace NaN with 0
        product['imputedsales'] = product['imputedsales'].fillna(0)
        product['minpurchase'] = product['minpurchase'].fillna(0)

        # sum product['imputedsales'] and product['minpurchase']
        product['minpurchase'] = product['imputedsales'] + product['minpurchase']

        # replace NaN in 'rating' and 'ovsentiment' with default/defined value
        product['rating'] = product['rating'].fillna(rating)
        product['ovsentiment'] = product['ovsentiment'].fillna(ovsentiment)

        # we only need 'imputeddate', 'id', and 'minpurchase'
        product = product[['id', 'imputeddate', 'minpurchase', 'rating', 'ovsentiment']]
        # rename columns name
        product.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']

        # convert sales to int
        product['Sales'] = product['Sales'].astype('int')

        # after all the processes above, return product
        return product

    else:
        pass

    product = product[['id', 'date', 'minpurchase', 'rating', 'ovsentiment']]
    product.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
    return product

def rightFormat(dataframe, freq):
    '''
    Simplifying the process of getting the right format in tsSalesRateSent function.
    dataframe: dataframe as input
    freq: frequency, a string -> 'D', 'W', 'M'

    the right format:
    'date'(as index) 'Sales'(sum) 'rating'(mean) 'ovsentiment'(mean)
    '''
    # do what should be done
    sales = dataframe[['date', 'Sales']].groupby(pd.Grouper(key='date', freq=freq)).sum()
    rating = dataframe[['date', 'rating']].groupby(pd.Grouper(key='date', freq=freq)).mean()
    ovsentiment = dataframe[['date', 'ovsentiment']].groupby(pd.Grouper(key='date', freq=freq)).mean()
    # join
    product = sales.join([rating, ovsentiment])
    product['rating'] = product['rating'].fillna(0)
    product['ovsentiment'] = product['ovsentiment'].fillna(0)
    # return product dataframe
    return product

def tsSalesRateSent(dataframe, freq='daily', standardize=False):
    '''
    --> time series_Sales-Rating-Sentiment

    Standardize all the data first.
    return a dataframe with the following criteria:
        Sales -> sum/cumsum per day/weekly/monthly
        rating -> average per day/weekly/monthly
        ovsentiment -> average per day/weekly/monthly
    dataframe columns name in order:
        'id' 'date' 'Sales' 'rating' 'ovsentiment'

    dataframe must contains the right 'date' format
    freq = a string. supports only 'daily', 'weekly', and 'monthly'
    '''
    dataframe.index = range(0, len(dataframe))
    mongoid = dataframe.loc[0, 'id']
    dataframe.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']

    if freq == 'daily':
        product = rightFormat(dataframe, freq='D')
    elif freq == 'weekly':
        product = rightFormat(dataframe, freq='W')
    elif freq == 'monthly':
        product = rightFormat(dataframe, freq='M')
    else:
        print('Not supported')
        print('The function supports only \'daily\', \'weekly\', and \'monthly\'')

    if standardize:
        # standardize 
        product = product.apply(zscore)
        return product
    else:
        # return 'product' dataframe
        return product

def granger(dataframe, maxlag=1, nonparametric=False):
    '''
    call grangertest function from R package, lmtest.
    run compiled c code.
    dataframe format:
        'date'(as index) 'Sales' 'rating' 'ovsentiment'

    '''
    # The Null hypothesis for grangercausalitytests is that 
    # the time series in the second column, x2, does NOT Granger cause 
    # the time series in the first column, x1.

    if nonparametric:
        # execute compiled C code
        pass
    else:
        # columns' names
        columns = tuple(dataframe.columns)

        for i in range(len(columns)):
            for j in range(len(columns)):
                if columns[j] != columns[i]:
                    xandy = columns[i] + ' -> ' + columns[j]
                    print(xandy)
                    df = dataframe[[columns[i], columns[j]]]
                    df = df.iloc[:, ].values
                    result = grangercausalitytests(df, maxlag=maxlag, verbose=True)
                    print('\n')

def terasvirta(dataframe, lag=1):
    '''
    Nonlinearity test
    '''
    pandas2ri.activate()
    # sort dataframe
    df = dataframe.sort_index()
    # R packages needed
    base = importr('base')
    tseries = importr('tseries')
    rxts = importr('xts', on_conflict="warn")

    # columns' names
    columns = df.columns
    # save terasvirta's p-value in the list below
    result_terasvirta = []
    # x ~ y
    names_terasvirta = []

    # Do terasvirta test for each columns against another
    # 'Sales' 'rating' 'ovsentiment'
    # 'Sales' ~ 'rating and 'Sales' ~ 'ovsentiment'
    # and so on
    for i in columns:
        x = rxts.as_xts(df[i])
        for j in columns[columns != i]:
            y = rxts.as_xts(df[j])
            # x~y | columns name
            xandy = i + ' ->  ' + j
            names_terasvirta.append(xandy)
            # result
            result = tseries.terasvirta_test(x, y)
            result_terasvirta.append(result[2][0])

    for i in range(len(result_terasvirta)):
        print('{0}; pvalue: {1}'.format(names_terasvirta[i], result_terasvirta[i]))

# Buat korelasi dan causality antara sales-sentiment, sales-ranking, sales-rating
# correlation & p-value:
# scipy.stats.pearsonr, scipy.stats.spearmanr, scipy.stats.kendalltau 

'''
# example
productlist = selectProd(conn, c, 20)
product = selectReview(productlist.loc[3, 'id'], conn, c, impute=True)
tsPlot(product, freq='weekly')
plt.ylabel('Sales')
plt.show()
tsPlot(product, freq='monthly', cumsum=True)
plt.show()

# example
ts = tsPlot(product, freq='weekly', result='df')
ts_lagged = ts.shift()
plt.plot(ts, color='blue')
plt.plot(ts_lagged, color='red')
plt.show()

# example ACF
from statsmodels.tsa import stattools

# example_ granger causality test
productlist = selectProd(conn, c, 20)
product = selectReview2('5aa274ad8cbad96d9a0c50be', conn, c)
test = tsSalesRateSent(product, freq='weekly')
granger(test, maxlag=2)
terasvirta(test)
'''

# check http://authorearnings.com/report/january-2015-author-earnings-report/
# check https://stackoverflow.com/questions/36105701/how-to-run-auto-arima-using-rpy2
'''
https://www.tokopedia.com/bantuan/overview-statistik-toko/
Tingkat konversi:
Asumsi:
jumlah transaksi sukses = minimal pembelian * reviewcount
turnover rate = prodseen / jumlah transaksi sukses
'''
