# library
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import math
from scipy.stats import zscore
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import acf, pacf, adfuller
import subprocess
# modules for terasvirta test
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import FloatVector
# r package 'forecast'

np.random.seed(42)

class sqlToDf:
    '''
    Create proper time series dataframe for time series modelling
    '''

    def __init__(self, connection, cursor):
        self.conn = connection
        self.c = cursor


        np.random.seed(42)

    def bla():
        pass
#          '''
#         Select product from prodpage table with actualrevcount => certain amount
#             inputted in by user.
#         returns dataframe.
#         actualrevcount: integer.
#         '''
#         sqlc = 'SELECT id, prodsold, salescluster, actualrevcount FROM prodpage WHERE actualrevcount >= '
#         sqlcall = sqlc + str(self.revcount)
#         self.c.execute(sqlcall)
#         self.conn.commit()
#     
#         self.prod = self.c.fetchall()
#         self.prod = pd.DataFrame(self.prod)
#         self.prod.columns = ['id', 'prodsold', 'salescluster', 'actualrevcount']       

    def selectReview(self, mongoid, impute=False):
        '''
        Select rowid, id, datereview, timereview
            based on mongoid and return dataframe
        mongoid: must be string. mongodb id.
        '''
        mongoid = '"' + mongoid + '"'
        sqlc = 'SELECT rowid, id, datereview from reviews WHERE id = '
        sqlc = sqlc + mongoid
        # execute
        self.c.execute(sqlc)
        self.conn.commit()
    
        # get all the query result and turn it into dataframe
        product = self.c.fetchall()
        product = pd.DataFrame(product)
        product.columns = ['rowid', 'id', 'date']
    
        # change string to date format
        product['date'] = pd.to_datetime(product['date'], format='%Y-%m-%d')
    
        # select record before march 01 2018
        product = product[product['date'] < '2018-03-01']
    
        # Minimal Purchase Quantity, product sold,
        # and actual review count
        sqlc1 = 'SELECT prodsold, minbeli, actualrevcount FROM prodpage WHERE id = ' + mongoid
        self.c.execute(sqlc1)
        self.conn.commit()
    
        minbeli = self.c.fetchall()
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
            "product.shape[0]" not daterange
            --> generate data sepanjang dataframe product
            test = np.random.multinomial(sumneeded, [1/product.shape[0].]*product.shape[0], 
                size=1).flatten()
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
            imputed = np.random.multinomial(sumneeded, 
                    [1/float(product.shape[0])]*product.shape[0], size=1).flatten()
            imputed = pd.DataFrame(imputed)
            imputed.columns = ['imputedsales']
            imputed['imputedsales'] = imputed['imputedsales'] * minbeli['minbeli'].values[0]
            imputed['imputeddate'] = pd.Series(pd.date_range(start=startdate, periods=daterange))

            # 
            product['imputedsales'] = imputed['imputedsales']
            # set new index
            product.index = range(0, len(product))
    
            # replace NaN with 0
            product['imputedsales'] = product['imputedsales'].fillna(0)
            product['minpurchase'] = product['minpurchase'].fillna(0)
    
            # sum product['imputedsales'] and product['minpurchase']
            product['sales'] = product['imputedsales'] + product['minpurchase']
    
            # we only need 'imputeddate', 'id', and 'minpurchase'
            product = product[['id', 'imputeddate', 'sales']]
            # rename columns name
            product.columns = ['id', 'date', 'Sales']
    
            # convert sales to int
            #product['Sales'] = product['Sales'].astype('int')
    
            # after all the processes above, return product
            return product
    
        else:
            pass
    
        product = product[['id', 'date', 'minpurchase']]
        product.columns = ['id', 'date', 'Sales']
        return product


    def selectReview2(self, mongoid, impute=False, rating=0, ovsentiment=0):
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
        self.c.execute(sqlc)
        self.conn.commit()
    
        # get all the query result and turn it into dataframe
        product = self.c.fetchall()
        product = pd.DataFrame(product)
        product.columns = ['rowid', 'id', 'date', 'rating', 'ovsentiment']
    
        # detect date format
        product['date'] = pd.to_datetime(product['date'], format='%Y-%m-%d')
    
        # select record before march 01 2018
        product = product[product['date'] < '2018-03-01']
    
        # Minimal Purchase Quantity, product sold, 
        # and actual review count
        sqlc1 = 'SELECT prodsold, minbeli, actualrevcount FROM prodpage WHERE id = ' + mongoid
        self.c.execute(sqlc1)
        self.conn.commit()
    
        minbeli = self.c.fetchall()
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
            imputed = np.random.multinomial(sumneeded, 
                    [1/float(product.shape[0])]*product.shape[0], size=1).flatten()
            imputed = pd.DataFrame(imputed)
            imputed.columns = ['imputedsales']
            imputed['imputedsales'] = imputed['imputedsales'] * minbeli.loc[0, 'minbeli']
            imputed['imputeddate'] = pd.Series(pd.date_range(start=startdate, periods=daterange))
            # add 'sales' in product with 'imputedsales'
            product['imputedsales'] = imputed['imputedsales']
            # set new index
            product.index = range(0, len(product))
    
            # replace NaN with 0
            product['imputedsales'] = product['imputedsales'].fillna(0)
            product['minpurchase'] = product['minpurchase'].fillna(0)

            # sum product['imputedsales'] and minbeli.loc[0, 'minbeli']
            product['sales'] = product['imputedsales'] + minbeli.loc[0, 'minbeli']
    
            # replace NaN in 'rating' and 'ovsentiment' with default/defined value
            product['rating'] = product['rating'].fillna(rating)
            product['ovsentiment'] = product['ovsentiment'].fillna(ovsentiment)
    
            # we only need 'imputeddate', 'id', and 'Sales'
            product = product[['id', 'date', 'sales', 'rating', 'ovsentiment']]
            # rename columns name
            product.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
    
            # after all the processes above, return product
            return product
    
        else:
            pass
    
        product = product[['id', 'date', 'minpurchase', 'rating', 'ovsentiment']]
        product.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
        return product

    def selectReview3(self, mongoid, impute=False, rating=0, ovsentiment=0):
        '''
        selectReview with sentiment and rating.
        Select rowid, id, datereview, timereview
            based on mongoid and return dataframe
        mongoid: must be string. mongodb id.
        rating: if impute=True and there is no review, default rating is 0
        ovsentiment: if impute=True and buyer didn't leave review, 
            default overall sentiment is neutral/0 and rating is 0
        '''
        mongoid = '"' + mongoid + '"'
        sqlc = 'SELECT rowid, id, datereview, rating, ovsentiment from reviews WHERE id = '
        sqlc = sqlc + mongoid
        # execute
        self.c.execute(sqlc)
        self.conn.commit()
    
        # get all the query result and turn it into dataframe
        product = self.c.fetchall()
        product = pd.DataFrame(product)
        product.columns = ['rowid', 'id', 'date', 'rating', 'ovsentiment']
    
        # detect date format
        product['date'] = pd.to_datetime(product['date'], format='%Y-%m-%d')
    
        # select record before march 01 2018
        product = product[product['date'] < '2018-03-01']
    
        # Minimal Purchase Quantity, product sold, 
        # and actual review count
        sqlc1 = 'SELECT prodsold, minbeli, actualrevcount FROM prodpage WHERE id = ' + mongoid
        self.c.execute(sqlc1)
        self.conn.commit()
    
        minbeli = self.c.fetchall()
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
            imputed = np.random.multinomial(sumneeded, 
                    [1/float(daterange)]*daterange, size=1).flatten()
            imputed = pd.DataFrame(imputed)
            imputed.columns = ['imputedsales']
            imputed['imputedsales'] = imputed['imputedsales'] * minbeli.loc[0, 'minbeli']
            imputed['imputeddate'] = pd.Series(pd.date_range(start=startdate, periods=(daterange+1)))

            # group 'product' dataframe by 'date' and 'rating', daily
            rating2 = product[['date', 'rating']].groupby(pd.Grouper(key='date', freq='D')).mean()
            # group 'product' dataframe by 'date' and 'ovsentiment', daily
            ovsentiment2 = product[['date', 'ovsentiment']].groupby(pd.Grouper(key='date', freq='D')).mean()
            # replace NaN with specified value
            rating2['rating'] = rating2['rating'].fillna(rating)
            ovsentiment2['ovsentiment'] = ovsentiment2['ovsentiment'].fillna(ovsentiment)
            # insert rating and ovsentiment into 'imputed' dataframe
            imputed['rating'] = rating2.reset_index()['rating']
            imputed['ovsentiment'] = ovsentiment2.reset_index()['ovsentiment']
            # add 'id' into 'imputed' dataframe
            product.index = range(0, len(product))
            imputed['id'] = product.loc[0, 'id']
            # rename 'imputed' dataframe columns
            imputed = imputed[['id', 'imputeddate', 'imputedsales', 'rating', 'ovsentiment']]
            imputed.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
            # set new index
            imputed.index = range(0, len(imputed))

            # after all the processes above, return product
            return imputed
    
        else:
            pass
    
        product = product[['id', 'date', 'minpurchase', 'rating', 'ovsentiment']]
        product.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
        product.index = range(0, len(product))

        # group 'product' dataframe by 'date' and 'Sales', daily
        sales3 = product[['date', 'Sales']].groupby(pd.Grouper(key='date', freq='D')).sum()
        # group 'product' dataframe by 'date' and 'rating', daily
        rating3 = product[['date', 'rating']].groupby(pd.Grouper(key='date', freq='D')).mean()
        # group 'product' dataframe by 'date' and 'ovsentiment', daily
        ovsentiment3 = product[['date', 'ovsentiment']].groupby(pd.Grouper(key='date', freq='D')).mean()
        # join
        temporary = sales3.join([rating3, ovsentiment3])
        temporary['rating'] = temporary['rating'].fillna(0)
        temporary['ovsentiment'] = temporary['ovsentiment'].fillna(0)
        # reset index
        temporary = temporary.reset_index()
        # add 'id'
        temporary['id'] = product.loc[0, 'id']
        # reorder temporary columns
        temporary = temporary[['id', 'date', 'Sales', 'rating', 'ovsentiment']]

        return temporary

    def selectReviewVal(self, mongoid, impute=False, rating=0, ovsentiment=0, notAll=True):
        '''
        selectReview with sentiment and rating.
        Select rowid, id, datereview, timereview
            based on mongoid and return dataframe
        mongoid: must be string. mongodb id.
        rating: if impute=True and there is no review, default rating is 0
        ovsentiment: if impute=True and buyer didn't leave review, 
            default overall sentiment is neutral/0
        notAll: if False, return all data from the date it was posted.
                if True, return data >= march 1 2018
        '''
        mongoid = '"' + mongoid + '"'
        sqlc = 'SELECT rowid, id, datereview, rating, ovsentiment from reviews WHERE id = '
        sqlc = sqlc + mongoid
        # execute
        self.c.execute(sqlc)
        self.conn.commit()
    
        # get all the query result and turn it into dataframe
        product = self.c.fetchall()
        product = pd.DataFrame(product)
        product.columns = ['rowid', 'id', 'date', 'rating', 'ovsentiment']
    
        # detect date format
        # '26 Jun 2018'
        product['date'] = pd.to_datetime(product['date'], format='%d %b %Y')
    
    
        # Minimal Purchase Quantity, product sold, 
        # and actual review count
        sqlc1 = 'SELECT prodsold, minbeli, actualrevcount FROM prodpage WHERE id = ' + mongoid
        self.c.execute(sqlc1)
        self.conn.commit()
    
        minbeli = self.c.fetchall()
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
            imputed = np.random.multinomial(sumneeded, 
                    [1/float(daterange)]*daterange, size=1).flatten()
            imputed = pd.DataFrame(imputed)
            imputed.columns = ['imputedsales']
            imputed['imputedsales'] = imputed['imputedsales'] * minbeli.loc[0, 'minbeli']
            imputed['imputeddate'] = pd.Series(pd.date_range(start=startdate, periods=(daterange+1)))

            # group 'product' dataframe by 'date' and 'rating', daily
            rating2 = product[['date', 'rating']].groupby(pd.Grouper(key='date', freq='D')).mean()
            # group 'product' dataframe by 'date' and 'ovsentiment', daily
            ovsentiment2 = product[['date', 'ovsentiment']].groupby(pd.Grouper(key='date', freq='D')).mean()
            # replace NaN with specified value
            rating2['rating'] = rating2['rating'].fillna(rating)
            ovsentiment2['ovsentiment'] = ovsentiment2['ovsentiment'].fillna(ovsentiment)
            # insert rating and ovsentiment into 'imputed' dataframe
            imputed['rating'] = rating2.reset_index()['rating']
            imputed['ovsentiment'] = ovsentiment2.reset_index()['ovsentiment']
            # add 'id' into 'imputed' dataframe
            product.index = range(0, len(product))
            imputed['id'] = product.loc[0, 'id']
            # rename 'imputed' dataframe columns
            imputed = imputed[['id', 'imputeddate', 'imputedsales', 'rating', 'ovsentiment']]
            imputed.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
            # set new index
            imputed.index = range(0, len(imputed))
            # select record after march 01 2018
            imputed = imputed[imputed['date'] >= '2018-03-01']

            # after all the processes above, return product
            return imputed
    
        else:
            pass
    
        product = product[['id', 'date', 'minpurchase', 'rating', 'ovsentiment']]
        product.columns = ['id', 'date', 'Sales', 'rating', 'ovsentiment']
        product.index = range(0, len(product))

        # group 'product' dataframe by 'date' and 'Sales', daily
        sales3 = product[['date', 'Sales']].groupby(pd.Grouper(key='date', freq='D')).sum()
        # group 'product' dataframe by 'date' and 'rating', daily
        rating3 = product[['date', 'rating']].groupby(pd.Grouper(key='date', freq='D')).mean()
        # group 'product' dataframe by 'date' and 'ovsentiment', daily
        ovsentiment3 = product[['date', 'ovsentiment']].groupby(pd.Grouper(key='date', freq='D')).mean()
        # join
        temporary = sales3.join([rating3, ovsentiment3])
        temporary['rating'] = temporary['rating'].fillna(0)
        temporary['ovsentiment'] = temporary['ovsentiment'].fillna(0)
        # reset index
        temporary = temporary.reset_index()
        # add 'id'
        temporary['id'] = product.loc[0, 'id']
        # reorder temporary columns
        temporary = temporary[['id', 'date', 'Sales', 'rating', 'ovsentiment']]

        # select record after march 01 2018
        if notAll:
            temporary = temporary[temporary['date'] >= '2018-03-01']
        else:
            pass

        return temporary

class tsTest:
    '''
    Time series test
    '''

    def __init__(self, dataframe, mongoid):
        self.df = dataframe
        self.mongoid = mongoid

    def granger(self, maxlag=1, nonparametric=False, folder='./gcttest/', embdim='2', bandwidth='1.5'):
        '''
        1 - call grangertest function from R package, lmtest.
        2 - run compiled c code for nonparametric causality.
        dataframe format:
            'date'(as index) 'Sales' 'rating' 'ovsentiment'
        folder: folder to store nonparametric files
        embdim: embedded dimension
        bandwidth: 
        '''
        # The Null hypothesis for grangercausalitytests is that 
        # the time series in the second column, x2, does NOT Granger cause 
        # the time series in the first column, x1.
    
        if nonparametric:
            # execute compiled C code

            # Also print mongoid for easy product identification
            print(self.mongoid)

            # columns' names
            columns = self.df.columns

            for i in range(len(columns)):
                for j in range(len(columns)):
                    if columns[j] != columns[i]:
                        xandy = columns[i] + ' -> ' + columns[j]
                        # save time series as .txt file without header and index
                        # path: './gctest/'
                        self.df[columns[i]].to_csv(folder+columns[i]+'.txt', header=None, 
                                index=None)
                        self.df[columns[j]].to_csv(folder+columns[j]+'.txt', header=None, 
                                index=None)
                        # file to store the result of non parametric granger test
                        # e.g.: 5a95d7ae35d6d33d3fea56ff_sales_rating.txt
                        outfile = folder + self.mongoid + '_' + columns[i] + '_' + columns[j] + '.txt'
                        # Execute c-compiled application with parameter:
                        # GCTtest inputfile1 inputfile2 embdim bandwidth outfile
                        # ./gcttest/GCTtest ./gcttest/sales.txt ./gcttest/rating.txt 2 1.5 
                        #           outfile
                        subprocess.call([folder + 'GCTtest', folder+columns[i]+'.txt',
                            folder+columns[j]+'.txt', embdim, bandwidth, outfile])
                        print(xandy)

        else:
            # columns' names
            columns = self.df.columns

            # Do granger test for each columns against another
            # 'Sales' 'rating' 'ovsentiment'
            # 'Sales' ~ 'rating and 'Sales' ~ 'ovsentiment'
            # and so on

            # Also print mongoid for easy product identification
            print(self.mongoid)

            for i in range(len(columns)):
                for j in range(len(columns)):
                    if columns[j] != columns[i]:
                        xandy = columns[i] + ' -> ' + columns[j]
                        print(xandy)
                        df = self.df[[columns[i], columns[j]]]
                        df = df.iloc[:, ].values
                        result = grangercausalitytests(df, maxlag=maxlag, verbose=True)
                        print('\n')

    def terasvirta(self, lag=1):
        '''
        Nonlinearity test
        '''
        pandas2ri.activate()
        # sort dataframe
        df = self.df.sort_index()
        # R packages needed
        base = importr('base')
        tseries = importr('tseries')
        rxts = importr('xts', on_conflict="warn", 
                robject_translations = {
                    ".subset.xts": "_subset_xts2"
                    }
                )
    
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

        # Also print mongoid for easy product identification
        print(self.mongoid)

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

    def adfuller(self, autolag = 'AIC', regression = 'ct'):
        '''
        Augmented Dickey-Fuller test
        self.df : a pandas Series of one column and datetime as index.
        '''
        useful_values_raw = adfuller(self.df, autolag='AIC', regression='ct')[:5]
        useful_values = [v for v in useful_values_raw[:4]]
        useful_values.extend([useful_values_raw[4]['1%'], 
            useful_values_raw[4]['5%'], useful_values_raw[4]['10%']])
        result = pd.DataFrame({
            'Value':useful_values, 
            'Label':['Test Statistic','p-value','#Lags Used',
                'Number of Observations Used', 'Critical value for 1%', 
                'Critical value for 5%', 'Critical value for 10%']
            })
        print(self.mongoid)
        return result

class neededPlots:
    '''
    Plots frequently used in time series analysis
    '''

    def __init__(self, dataframe, mongoid):
        self.df = dataframe
        self.mongoid = mongoid

    def acfLine(self, save=False, nlags=40, imgdir=None):
        '''
            ACF (Auto-Correlation Function) for Moving Average
        model diagnostics.
        save : Boolean. whether save the plot generated or not.
        nlags : an integer. number of lags desired.
        imgdir : a string. image directory to store saved plot.
        '''
        
        if save:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # acf
            acf_result = acf(diff_df, nlags=nlags)
            # plot acf
            plt.plot(acf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - ACF Plot'
            plt.title(fortitle)
            plt.savefig(imgdir, format='eps', dpi=1000)
        else:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # acf
            acf_result = acf(diff_df, nlags=nlags)
            # plot acf
            plt.plot(acf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - ACF Plot'
            plt.title(fortitle)
            plt.show()

    def pacfLine(self, save=False, nlags=40, imgdir=None):
        '''
            PACF (Partial Auto-Correlation Function) for AutoRegressive
        model diagnostics
        '''

        if save:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # pacf
            pacf_result = pacf(diff_df, nlags=nlags)
            # plot acf
            plt.plot(pacf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - PACF Plot'
            plt.title(fortitle)
            plt.savefig(imgdir, format='eps', dpi=1000)
        else:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # pacf
            pacf_result = pacf(diff_df, nlags=nlags)
            # plot acf
            plt.plot(pacf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - PACF Plot'
            plt.title(fortitle)
            plt.show()

    def acfBar(self, save=False, nlags=40, imgdir=None):
        '''
        This function create a bar plot for ACF.
            ACF (Auto-Correlation Function) for Moving Average
        model diagnostics.
        save : Boolean. whether save the plot generated or not.
        nlags : an integer. number of lags desired.
        imgdir : a string. image directory to store saved plot.
        '''
        
        if save:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # acf
            acf_result = acf(diff_df, nlags=nlags)
            # plot acf
            plt.bar(left = range(len(acf_result)), height = acf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - ACF Plot'
            plt.title(fortitle)
            plt.savefig(imgdir, format='eps', dpi=1000)
        else:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # acf
            acf_result = acf(diff_df, nlags=nlags)
            # plot acf
            plt.bar(left = range(len(acf_result)), height = acf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - ACF Plot'
            plt.title(fortitle)
            plt.show()

    def pacfBar(self, save=False, nlags=40, imgdir=None):
        '''
        This function create a bar plot for PACF.
            PACF (Partial Auto-Correlation Function) for AutoRegressive
        model diagnostics
        '''

        if save:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # pacf
            pacf_result = pacf(diff_df, nlags=nlags)
            # plot acf
            plt.bar(left = range(len(pacf_result)), height = pacf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - PACF Plot'
            plt.title(fortitle)
            plt.savefig(imgdir, format='eps', dpi=1000)
        else:
            # differencing
            diff_df = self.df - self.df.shift()
            diff_df = diff_df.dropna()
            # pacf
            pacf_result = pacf(diff_df, nlags=nlags)
            # plot acf
            plt.bar(left = range(len(pacf_result)), height = pacf_result)
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--')
            plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--')
            fortitle = self.mongoid + ' - PACF Plot'
            plt.title(fortitle)
            plt.show()

######### END OF CLASSES #########
##################################

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

def tsSalesRateSentiment(dataframe, freq='daily', standardize=False):
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
#        product = rightFormat(dataframe, freq='D')
        product = dataframe.copy()
        product = product[['date', 'Sales', 'rating', 'ovsentiment']]
        product = product.set_index('date')
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

def showClusterData(conn, cursor, clust, sortby):
    '''
    This function shows actual review count and ranking based on cluster
    specified.
    conn: sqlite3 connection
    cursor: sqlite3 cursor
    clust: int. cluster
    sortby: string or list of string.
    '''
    sqlc = 'SELECT id, ranking, actualrevcount, salescluster FROM prodpage WHERE salescluster == '
    clustnumber = str(float(clust))
    sqlcall = sqlc + clustnumber
    cursor.execute(sqlcall)
    conn.commit()
    # get all the query
    df = cursor.fetchall()
    # convert it into dataframe
    df = pd.DataFrame(df)
    df.columns = ['id', 'ranking', 'actualrevcount', 'salescluster']
    # sort by
    df = df.sort_values(by=sortby, ascending=False, na_position='first')
    return df

def genData(mongoid, conn, cursor, impute=True, freq='daily', actualrevcount=20):
    '''
    Generate a timeseries dataframe for timeseries modelling.
    mongoid: str. string of mongodb id.
    conn: sqlite3 connection.
    cursor: sqlite3 cursor.
    impute:
    freq:
    actualrevcount:
    '''
    initial = routines.sqlToDf(conn, cursor, actualrevcount=actualrevcount)
    allproduct = initial.selectReview2(mongoid, impute=impute)
    product = routines.tsSalesRateSentiment(allproduct, freq=freq)
    return product
    # product = genData('5aa2ad7735d6d34b0032a795', conn, c, impute=True,
    #   freq='daily', actualrevcount=20)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data.copy()
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg = agg.dropna()
    return agg

def splitDataNN(df, n_in=1, n_out=1, scale=True, percent=0.2):
    '''
    df: pandas dataframe. 3 columns (sales, rating, ovsentiment) with date as index
    n_in:
    n_out:
    scale:
    percent:
    X_train, y_train, X_test, y_test, dftrain = splitDataNN(product, n_in=1,
        n_out=1, scale=True, percent=0.2)
    '''
    dftrain = series_to_supervised(df, n_in=n_in, n_out=n_out)
    # specific to this case
    dftrain = dftrain.drop(dftrain.columns[[4, 5]], axis=1)
    values = dftrain.values

    if scale:
        scaler = MinMaxScaler()
        values = scaler.fit_transform(values)
    else:
        pass

    # training data
    X, y = values[:, :-1], values[:, -1]
    # train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent,
            shuffle=False, random_state=42)
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train, y_train, X_test, y_test, dftrain, scaler

def genDataVal(mongoid, conn, cursor, impute=False, freq='daily'):
    '''
    Generate a timeseries dataframe for timeseries modelling.
    mongoid: str. string of mongodb id.
    conn: sqlite3 connection.
    cursor: sqlite3 cursor.
    impute:
    freq:
    actualrevcount:
    '''
    # list of old id-s
    listOfId = ['5aa2ad7735d6d34b0032a795', '5aa39533ae1f941be7165ecd',
            '5aa2c35e35d6d34b0032a796', '5a93e8768cbad97881597597',
            '5a9347b98cbad97074cb1890']
    if mongoid in listOfId:
        sqlc = 'SELECT id, idProd FROM prodpage WHERE idProd = '
        old = '"' + mongoid + '"'
        sqlcall = sqlc + old
        cursor.execute(sqlcall)
        conn.commit()

        temp = cursor.fetchone()
        mongoid = temp[0]
    else:
        pass
#        print('Input new MongoDB ID!')

    initial = sqlToDf(conn, cursor)
    notts = initial.selectReviewVal(mongoid, impute=impute)
    product = tsSalesRateSentiment(notts, freq=freq)
    return product

######### END OF CLASSES AND FUNCTIONS #########
################################################

'''
import tseriesRoutines as routines
import matplotlib.pyplot as plt
import sqlite3

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

test = routines.sqlToDf(conn, c, 20)
test.prod.head()
product = test.selectReview2(test.prod.loc[3, 'id'], impute=True)
product2 = routines.tsSalesRateSentiment(product, freq='daily')

tstest = routines.tsTest(product2, product.loc[0,'id'])
tstest.granger()
tstest.granger(nonparametric=True)
tstest.terasvirta()
tstest.adfuller()

# acf and pacf
tsplots = routines.neededPlots(product2, test.prod.loc[3, 'id'])
tsplots.acfLine()
tsplots.pacfLine()
tsplots.acfBar()
tsplots.pacfBar()

product2 = routines.tsSalesRateSentiment(product, freq='weekly', standardize=True)
product2.plot()
plt.show()

df = routines.showClusterData(conn, c, 3, sortby=['ranking', 'actualrevcount'])
df1 = routines.showClusterData(conn, c, 1, sortby=['ranking', 'actualrevcount'])
df1_2 = df1[df1['actualrevcount'] >= 35]
df1_2.sort_values(by=['ranking']).head(10)
'''
