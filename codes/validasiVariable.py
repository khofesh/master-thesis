'''
membuat file csv untuk validasi model ranking

v 1. discount value
v 2. discount rate ~ cashback
v 3. current price
v 4. review valence
v 5. review volume
v 6. percentage of negative review
v 7. percentage of positive review
v 8. number of answered questions
v 9. number of people who find reviews helpful
v 10. rating of the most helpful positive review
v 11. positive sentiment strength
v 12. negative sentiment strength
v 13. sentiment polarity
v 14. reviewer ranking ~ diganti smiley/reviewers reputation
v 15. picture of reviewer
v 16. picture of products
v 17. sales rank
minimal pembelian
'''

import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm
import os
import cv2 as cv
import imghdr

def cashbackVal(connection, cursor, csvoutput):
    '''
    cashback value ~ discount value substitute
    '''
    sqlc = 'SELECT id, price, cashback, cashbackval FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'price', 'cashback', 'cashbackval']

    # cashback * price
    prodpage['cashbackval'] = prodpage['price'] * prodpage['cashback']

    # write to csv file
    prodpage.to_csv(csvoutput, index=False)

    # updating 'cashbackval' column in prodpage table
    sqlc1 = 'UPDATE prodpage SET '
    sqlc2 = 'cashbackval = '
    sqlc3 = 'WHERE id = '

    print('Updating \'cashbackval\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        cashbackval = str(prodpage.loc[i, 'cashbackval'])
        sqlall = sqlc1 + sqlc2 + cashbackval + ' ' + sqlc3 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def posNegReview(connection, cursor, csvoutput):
    '''
    Positive reviews:
        number of 5 and 4 star customer reviews divided by the
        total number of customer reviews.
    Negative reviews:
        number of 2 and 1 star customer reviews divided by the
        total number of customer reviews
    '''
    # Positive
    sqlc = 'SELECT id, rating FROM reviews WHERE rating == 5 OR rating == 4'
    cursor.execute(sqlc)
    connection.commit()
    reviews1 = cursor.fetchall()
    reviews1 = pd.DataFrame(reviews1)
    reviews1.columns = ['id', 'rating']
    reviews1['numpos'] = 1
    # groupby 'id'
    reviews1['id'] = reviews1['id'].astype('category')
    reviews1 = reviews1.groupby(['id']).sum().reset_index()

    # Negative
    sqlc1 = 'SELECT id, rating FROM reviews WHERE rating == 1 OR rating == 2'
    cursor.execute(sqlc1)
    connection.commit()
    reviews2 = cursor.fetchall()
    reviews2 = pd.DataFrame(reviews2)
    reviews2.columns = ['id', 'rating']
    reviews2['numneg'] = 1
    # groupby 'id'
    reviews2['id'] = reviews2['id'].astype('category')
    reviews2 = reviews2.groupby(['id']).sum().reset_index()

    # merge
    reviews = pd.merge(reviews1, reviews2, on='id', how='outer')
    # replace NaN with 0
    reviews['numneg'] = reviews['numneg'].fillna(0)
    reviews['numneg'] = reviews['numneg'].astype('int')

    # posreview
    reviews['posreview'] = reviews['numpos'] / (reviews['numpos'] + reviews['numneg'])
    # negreview
    reviews['negreview'] = reviews['numneg'] / (reviews['numpos'] + reviews['numneg'])

    # reviews all
    reviews = reviews[['id', 'numpos', 'numneg', 'posreview', 'negreview']]

    # write to csv file
    reviews.to_csv(csvoutput, index=False)

    # updating 'posreview' column in prodpage table
    sqlc3 = 'UPDATE prodpage SET '
    sqlc4 = 'posreview = '
    sqlc5 = 'WHERE id = '

    print('Updating \'posreview\' column')
    for i in tqdm(range(len(reviews))):
        mongoid = reviews.loc[i, 'id']
        posreview = str(reviews.loc[i, 'posreview'])
        sqlall = sqlc3 + sqlc4 + posreview + ' ' + sqlc5 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

    # updating 'negreview' column in prodpage table
    sqlc6 = 'negreview = '

    print('Updating \'negreview\' column')
    for i in tqdm(range(len(reviews))):
        mongoid = reviews.loc[i, 'id']
        negreview = str(reviews.loc[i, 'negreview'])
        sqlall = sqlc3 + sqlc6 + negreview + ' ' + sqlc5 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def otherAgree(connection, cursor, csvoutput):
    '''
    number of people who find reviews helpful
    '''
    # prodpage
    sqlc = 'SELECT id, otheragreemean FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'otheragreemean']
    prodpage['id'] = prodpage['id'].astype('category')

    # reviews
    sqlc1 = 'SELECT rowid, id, otheragree FROM reviews'
    cursor.execute(sqlc1)
    connection.commit()
    reviews = cursor.fetchall()
    reviews = pd.DataFrame(reviews)
    reviews.columns = ['rowid', 'id', 'otheragree']
    reviews['id'] = reviews['id'].astype('category')
    # groupby 'id'
    reviews = reviews.groupby(['id']).mean()
    reviews = reviews.reset_index()
    reviews['otheragree'] = reviews['otheragree'].astype('int')

    # merge
    prodpage = pd.merge(prodpage, reviews, on='id', how='outer')
    prodpage['otheragreemean'] = prodpage['otheragree']
    prodpage = prodpage[['id', 'otheragreemean']]
    prodpage['otheragreemean'] = prodpage['otheragreemean'].fillna(0)
    prodpage['otheragreemean'] = prodpage['otheragreemean'].astype('int')

    # save to csv file
    prodpage.to_csv(csvoutput, index=False)

    # updating 'otheragreemean' column in prodpage table
    sqlc2 = 'UPDATE prodpage SET '
    sqlc3 = 'otheragreemean = '
    sqlc4 = 'WHERE id = '

    print('Updating \'otheragreemean\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        otheragreemean = str(prodpage.loc[i, 'otheragreemean'])
        sqlall = sqlc2 + sqlc3 + otheragreemean + ' ' + sqlc4 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def ratingMostHelpful(connection, cursor, csvoutput):
    '''
    Rating of the most helpful positive review:
        star rating of the positive review of a product
        with the highest agreement on helpfulness
    '''
    # prodpage
    sqlc = 'SELECT id, ratingmosthelpful FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'ratingmosthelpful']
    prodpage['id'] = prodpage['id'].astype('category')

    # reviews
    sqlc1 = 'SELECT rowid, id, rating FROM reviews WHERE helpful == 1'
    cursor.execute(sqlc1)
    connection.commit()
    reviews = cursor.fetchall()
    reviews = pd.DataFrame(reviews)
    reviews.columns = ['rowid', 'id', 'rating']
    reviews['id'] = reviews['id'].astype('category')
    # groupby 'id'
    reviews = reviews.groupby(['id']).mean()
    reviews = reviews.reset_index()
    reviews['rating'] = reviews['rating'].astype('int')

    # merge
    prodpage = pd.merge(prodpage, reviews, on='id', how='outer')
    prodpage['ratingmosthelpful'] = prodpage['rating']
    prodpage = prodpage[['id', 'ratingmosthelpful']]
    prodpage['ratingmosthelpful'] = prodpage['ratingmosthelpful'].fillna(0)
    prodpage['ratingmosthelpful'] = prodpage['ratingmosthelpful'].astype('int')

    # save to csv file
    prodpage.to_csv(csvoutput, index=False)

    # updating 'ratingmosthelpful' column in prodpage table
    sqlc2 = 'UPDATE prodpage SET '
    sqlc3 = 'ratingmosthelpful = '
    sqlc4 = 'WHERE id = '

    print('Updating \'ratingmosthelpful\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        ratingmosthelpful = str(prodpage.loc[i, 'ratingmosthelpful'])
        sqlall = sqlc2 + sqlc3 + ratingmosthelpful + ' ' + sqlc4 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def reviewersRep(connection, cursor, csvoutput):
    '''
    Buyer reputation or reviewer's reputation.
    average reputation of reviewers.
    neutral smiley is not used.
    '''
    # prodpage
    sqlc = 'SELECT id, reviewersrep FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'reviewersrep']
    # id as category
    prodpage['id'] = prodpage['id'].astype('category')

    # reviews
    sqlc1 = 'SELECT rowid, id, negsmiley, possmiley FROM reviews'
    cursor.execute(sqlc1)
    connection.commit()
    reviews = cursor.fetchall()
    reviews = pd.DataFrame(reviews)
    reviews.columns = ['rowid', 'id', 'negsmiley', 'possmiley']
    reviews['reputation'] = 0
    # if reviews['negsmiley'] < reviews['possmiley'] -> 1
    # if reviews['negsmiley'] > reviews['possmiley'] -> -1
    # if reviews[
    mask = reviews['negsmiley'] < reviews['possmiley']
    mask2 = reviews['negsmiley'] > reviews['possmiley']
    reviews.loc[mask, 'reputation'] = 1
    reviews.loc[mask2, 'reputation'] = -1
    reviews['reputation'].replace(0, 1)
    # id as category
    reviews['id'] = reviews['id'].astype('category')
    # groupby id, then average 'reputation'
    reviews = reviews.groupby(['id']).mean()
    # id not as index
    reviews = reviews.reset_index()
    # exclude 'rowid', 'negsmiley' and 'possmiley'
    reviews = reviews[['id', 'reputation']]
    # reputation as int
    reviews['reputation'] = reviews['reputation'].astype('int')
    # after looking into the data,
    # we decided reputation = 0 -> 1
    mask3 = reviews['reputation'] == 0
    reviews.loc[mask3, 'reputation'] = 1

    # merge
    prodpage = pd.merge(prodpage, reviews, on='id', how='outer')
    prodpage['reviewersrep'] = prodpage['reputation']
    prodpage = prodpage[['id', 'reviewersrep']]

    # write to csv file
    prodpage.to_csv(csvoutput, index=False)

    # update 'reviewersrep' in prodpage table
    print('Update \'reviewersrep\' column in prodpage table')
    sqlc2 = 'UPDATE prodpage SET '
    sqlc3 = 'reviewersrep = '
    sqlc4 = 'WHERE id = '

    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        reviewersrep = str(prodpage.loc[i, 'reviewersrep'])
        sqlall = sqlc2 + sqlc3 + reviewersrep + ' ' + sqlc4 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def compareImg(defavadir, picpath):
    '''
    '''
    # get default avatar directory path
    fullpath = os.path.join(os.getcwd(), defavadir)
    # list of default avatars
    listofdefav = os.listdir(fullpath)
    fullpathpics = [os.path.join(fullpath, pic) for pic in listofdefav]
    # convert picture into vector
    picvector = [cv.imread(pic) for pic in fullpathpics]

    # test whether image picpath is gif or not
    # if it is, then return 1
    if imghdr.what(picpath) == 'gif':
        return 1
    else:
        profpicvector = cv.imread(picpath)

    # subtract profpicvector with each of picvector
    # np.any will return True when the result of subtraction is not zero
    # and will return False when every element of subtraction result is zero
    try:
        # some picture file has different vector size, so sometimes
        # the following will get an error
        vecsubtraction = [np.any(cv.subtract(vec, profpicvector)) for vec in picvector]
    except cv.error:
        return 1
    # sum([True, True, True, True]) = 4
    if sum(vecsubtraction) == 4:
        # 1 means it's profile picture
        return 1
    else:
        # 0 means it's one of the default avatar
        return 0

def reviewerPic(connection, cursor, csvoutput, defavadir):
    '''
    Picture of reviewer:
        total number of reviewers with profile picture
    defavadir = default avatar directory path
    DELETE ALL .jpg FILES IN folder profpics which has size 0 byte:
        find ./profpics -size 0 -exec rm -rf {} \;
    OTHERWHISE, cv2.error
    '''
    path = os.getcwd()
    profpics = 'profpics'

    # get id from prodpage table
    sqlc = 'SELECT id, revpictotal FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'revpictotal']


    for i in tqdm(range(len(prodpage))):
        fullpath = os.path.join(path, profpics, prodpage.loc[i, 'id'])
        if not os.listdir(fullpath):
            prodpage.loc[i, 'revpictotal'] = 9999
        elif os.listdir(fullpath):
            listofpics = os.listdir(fullpath)
            fullpathpics = [os.path.join(fullpath, pic) for pic in listofpics]
            # default profpic or profpic uploaded by reviewer
            defaultornot = [compareImg(defavadir, picpath) for picpath in fullpathpics]
            prodpage.loc[i, 'revpictotal'] = sum(defaultornot)
        else:
            print('something is wrong')

    # write to csv file
    prodpage.to_csv(csvoutput, index=False)

    # update revpictotal column in product.db
    sqlc1 = 'UPDATE prodpage SET '
    sqlc2 = 'revpictotal = '
    sqlc3 = 'WHERE id = '

    print('Updating \'revpictotal\' column in prodpage table')
    for k in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[k, 'id']
        # must be string, not int
        revpictotal = str(prodpage.loc[k, 'revpictotal'])
        sqlall = sqlc1 + sqlc2 + revpictotal + ' ' + sqlc3 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def prodPics(connection, cursor, csvoutput):
    '''
    picture of products: total number of reviews with pictures
    '''
    sqlc = 'SELECT id, prodpicstotal FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'prodpicstotal']
    # convert 'id' type into category
    prodpage['id'] = prodpage['id'].astype('category')

    # prodpic in reviews table
    sqlc1 = 'SELECT rowid, id, prodpic FROM reviews'
    cursor.execute(sqlc1)
    connection.commit()
    reviews = cursor.fetchall()
    reviews = pd.DataFrame(reviews)
    reviews.columns = ['rowid', 'id', 'prodpic']
    # convert 'id' type into category
    reviews['id'] = reviews['id'].astype('category')
    # groupby 'id', then sum 'prodpic'
    reviews = reviews.groupby(['id']).sum()
    reviews = reviews.reset_index()

    # merge on 'id'
    prodpage = pd.merge(prodpage, reviews, on='id', how='outer')
    prodpage['prodpicstotal'] = prodpage['prodpic']
    prodpage = prodpage[['id', 'prodpicstotal']]
    prodpage['prodpicstotal'] = prodpage['prodpicstotal'].astype('int')

    # write to csv file
    prodpage.to_csv(csvoutput, index=False)

    # update 'prodpicstotal' column in prodpage table
    print('Update \'prodpicstotal\' column in prodpage table')
    sqlc2 = 'UPDATE prodpage SET '
    sqlc3 = 'prodpicstotal = '
    sqlc4 = 'WHERE id = '

    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        prodpicstotal = str(prodpage.loc[i, 'prodpicstotal'])
        sqlall = sqlc2 + sqlc3 + prodpicstotal + ' ' + sqlc4 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def salesRank(connection, cursor, csvoutput):
    # query product
    sqlc = '''SELECT
            id, prodname, prodseen, prodsold, reputation,
            reviewcount, ranking
            FROM prodpage'''
    cursor.execute(sqlc)
    connection.commit()

    product = cursor.fetchall()
    product = pd.DataFrame(product)
    product.columns = ['id', 'prodname', 'prodseen', 'prodsold',
            'reputation', 'reviewcount', 'ranking']

    # if prodsold = 0 or prodsold < reviewcount, set prodsold = reviewcount
    mask = product['prodsold'] < product['reviewcount']
    # get id to update 'prodsold' in sql database later
    dataLessOrZero = product[mask].copy()
    dataLessOrZero = dataLessOrZero.reset_index()
    dataLessOrZero = dataLessOrZero[['id', 'prodsold']]
    column_name = 'prodsold'
    product.loc[mask, column_name] = product.loc[mask, 'reviewcount']

    # sort by prodsold (descending)
    product = product.sort_values(by='prodsold', ascending=False)
    product = product.reset_index()

    # set ranking
    product['ranking'] = pd.Series(range(1, len(product)+1))

    # update 'prodsold' in sql db
    sqlc1 = 'UPDATE prodpage SET '
    sqlc2 = 'prodsold = '
    sqlc3 = 'WHERE id = '
    print('Update prodsold value in database')
    for i in tqdm(range(len(dataLessOrZero))):
        mongoid = dataLessOrZero.loc[i, 'id']
        prodsold = product[product['id'] == mongoid]['prodsold'].values[0]
        sqlall = sqlc1 + sqlc2 + str(prodsold) + ' ' + sqlc3 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

    # save to csv file
    print('Saving to csv file')
    product.to_csv(csvoutput, index=False)

    # update 'ranking' in sql db
    sqlc2 = 'ranking = '
    print('Update ranking value in database')
    for j in tqdm(range(len(product))):
        mongoid = product.loc[j, 'id']
        ranking = product.loc[j, 'ranking']
        sqlall = sqlc1 + sqlc2 + str(ranking) + ' ' + sqlc3 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def minPurchase(connection, cursor, csvinput, csvoutput):
    # read csv file
    minbeli = pd.read_csv(csvinput)

    # query
    sqlc = 'SELECT id, uri FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = ['id', 'uri']

    # merge
    minbeli = pd.merge(result, minbeli, on='uri', how='outer')
    # delete duplicate
    minbeli = minbeli[~minbeli.duplicated(keep='first')]
    # NaN -> 1
    mask = minbeli['minpurchase'].isna()
    minbeli.loc[mask, 'minpurchase'] = 1
    # convert float -> integer
    minbeli['minpurchase'] = minbeli['minpurchase'].astype('int')
    # reset index
    minbeli = minbeli.reset_index()

    # save to csv file
    print('Saving to csv file')
    minbeli.to_csv(csvoutput, index=False)

    # update product.db 'minbeli' column
    sqlc1 = 'UPDATE prodpage SET '
    sqlc2 = 'minbeli = '
    sqlc3 = 'WHERE id = '

    print('Updating sql column \'minbeli\'')
    for i in tqdm(range(len(minbeli))):
        mongoid = minbeli.loc[i, 'id']
        minpurchase = str(minbeli.loc[i, 'minpurchase'])
        sqlall = sqlc1 + sqlc2 + minpurchase + ' ' + sqlc3 + '"' + mongoid + '"'
#        print(sqlall)
        cursor.execute(sqlall)
        connection.commit()

def actualRevCount(connection, cursor, csvoutput):
    '''
    fill in the value of acturalrevcount column in prodpage table
    '''
    # prodpage
    sqlc = 'SELECT id, actualrevcount FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'actualrevcount']
    prodpage['id'] = prodpage['id'].astype('category')

    # reviews
    sqlc1 = 'SELECT rowid, id FROM reviews'
    cursor.execute(sqlc1)
    connection.commit()
    reviews = cursor.fetchall()
    reviews = pd.DataFrame(reviews)
    reviews.columns = ['rowid', 'id']
    reviews['id'] = reviews['id'].astype('category')
    # count
    reviews['count'] = 1
    # groupby 'id'
    reviews = reviews.groupby(['id']).sum()
    reviews = reviews.reset_index()
    reviews['count'] = reviews['count'].astype('int')

    # merge
    prodpage = pd.merge(prodpage, reviews, on='id', how='outer')
    prodpage['actualrevcount'] = prodpage['count']
    prodpage = prodpage[['id', 'actualrevcount']]

    # save to csv file
    prodpage.to_csv(csvoutput, index=False)

    # updating 'otheragreemean' column in prodpage table
    sqlc2 = 'UPDATE prodpage SET '
    sqlc3 = 'actualrevcount = '
    sqlc4 = 'WHERE id = '

    print('Updating \'actualrevcount\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        actualrevcount = str(prodpage.loc[i, 'actualrevcount'])
        sqlall = sqlc2 + sqlc3 + actualrevcount + ' ' + sqlc4 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def prodpageTrain(connection, cursor):
    '''
    create csv file for model training
    all variables:
        discount value = cashback
        discount rate = cashbackval
        current price = price
        review valence = prodrating
        review volume = reviewcount
        % of negative review = negreview
        % of positive review = posreview
        number of answered questions = answerscnt
        sales rank = ranking
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
    addendum:
        id, prodname, actualrevcount, merchantname
    '''
    # query
    sqlc = '''
        SELECT id, prodname, merchantname, merchanttype, topads,
            actualrevcount, cashback, cashbackval,
            price, prodrating, reviewcount, negreview, posreview, answerscnt,
            otheragreemean, ratingmosthelpful, possentiment, negsentiment,
            sentipolarity, reviewersrep, revpictotal, prodpicstotal, ranking
        FROM prodpage
    '''
    cursor.execute(sqlc)
    connection.commit()
    traindata = cursor.fetchall()
    traindata = pd.DataFrame(traindata)
    traindata.columns = ['id', 'prodname', 'merchantname', 'merchanttype', 'topads',
            'actualrevcount', 'cashback',
            'cashbackval','price', 'prodrating', 'reviewcount', 'negreview', 'posreview',
            'answerscnt', 'otheragreemean', 'ratingmosthelpful', 'possentiment',
            'negsentiment', 'sentipolarity', 'reviewersrep', 'revpictotal',
            'prodpicstotal', 'ranking']

    return traindata

def salesCluster(connection, cursor, csvcluster):
    '''
    Update salescluster value in prodpage table
    '''
    prodpage = pd.read_csv(csvcluster)

    # updating 'otheragreemean' column in prodpage table
    sqlc = 'UPDATE prodpage SET '
    sqlc1 = 'salescluster = '
    sqlc2 = 'WHERE id = '

    print('Updating \'salescluster\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        salescluster = str(prodpage.loc[i, 'cluster'])
        sqlall = sqlc + sqlc1 + salescluster + ' ' + sqlc2 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def meanRank(connection, cursor, csvoutput):
    '''
    Assign ranks to data, dealing with ties appropriately.
    method: average
    https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html#r357
    '''
    sqlc = 'SELECT id, prodsold, ranking FROM prodpage'
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'prodsold', 'ranking']
    prodpage['ranking'] = prodpage['ranking'].astype('float')

    from scipy.stats import rankdata
    # assign rank to prodsold
    prodpage['ranking'] = rankdata(prodpage['prodsold'])

    # write to csv file
    prodpage.to_csv(csvoutput, index=False)

    # updating 'ranking' column in prodpage table
    sqlc1 = 'UPDATE prodpage SET '
    sqlc2 = 'ranking = '
    sqlc3 = 'WHERE id = '

    print('Updating \'ranking\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        ranking = str(prodpage.loc[i, 'ranking'])
        sqlall = sqlc1 + sqlc2 + ranking + ' ' + sqlc3 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def updateProdsold(connection, cursor, csvoutput):
    '''
    update prodsold, whose value < actualrevcount.
    prodsold = actualrevcount
    '''

    sqlc = '''SELECT id, prodsold, actualrevcount FROM prodpage
        WHERE prodsold < actualrevcount'''
    cursor.execute(sqlc)
    connection.commit()
    prodpage = cursor.fetchall()
    prodpage = pd.DataFrame(prodpage)
    prodpage.columns = ['id', 'prodsold', 'actualrevcount']
    prodpage['actualrevcount'] = prodpage['actualrevcount'].astype('int')
    # prodsold = actualrevcount
    prodpage['prodsold'] = prodpage['actualrevcount']

    # write to csv file
    prodpage.to_csv(csvoutput, index=False)

    # updating 'prodsold' column in prodpage table
    sqlc1 = 'UPDATE prodpage SET '
    sqlc2 = 'prodsold = '
    sqlc3 = 'WHERE id = '

    print('Updating \'prodsold\' column')
    for i in tqdm(range(len(prodpage))):
        mongoid = prodpage.loc[i, 'id']
        prodsold = str(prodpage.loc[i, 'prodsold'])
        sqlall = sqlc1 + sqlc2 + prodsold + ' ' + sqlc3 + '"' + mongoid + '"'
        cursor.execute(sqlall)
        connection.commit()

def trainingTable(connection, cursor):
    '''
    create table named 'training' in product database.
    '''
    # Create training table
    sqlc = '''
        CREATE TABLE IF NOT EXISTS training (
        id TEXT NOT NULL,
        prodname TEXT,
        merchantname TEXT,
        merchanttype TEXT,
        topads BOOLEAN,
        actualrevcount REAL,
        cashback REAL,
        cashbackval REAL,
        price INTEGER,
        prodrating REAL,
        reviewcount INTEGER,
        negreview REAL,
        posreview REAL,
        answerscnt INTEGER,
        otheragreemean TEXT,
        ratingmosthelpful INTEGER,
        possentiment INTEGER,
        negsentiment INTEGER,
        sentipolarity INTEGER,
        reviewersrep INTEGER,
        revpictotal INTEGER,
        prodpicstotal INTEGER,
        ranking INTEGER,
        FOREIGN KEY (id) REFERENCES prodpage(id)
            ON UPDATE CASCADE ON DELETE CASCADE
        )'''
    cursor.execute(sqlc)
    connection.commit()

def upTrainTable(connection, cursor, tablename):
    # query
    sqlc = '''
        SELECT id, prodname, merchantname, merchanttype, topads,
            actualrevcount, cashback, cashbackval,
            price, prodrating, reviewcount, negreview, posreview, answerscnt,
            otheragreemean, ratingmosthelpful, possentiment, negsentiment,
            sentipolarity, reviewersrep, revpictotal, prodpicstotal, ranking
        FROM prodpage
    '''
    cursor.execute(sqlc)
    connection.commit()
    traindata = cursor.fetchall()
    traindata = pd.DataFrame(traindata)
    traindata.columns = ['id', 'prodname', 'merchantname', 'merchanttype', 'topads',
            'actualrevcount', 'cashback',
            'cashbackval','price', 'prodrating', 'reviewcount', 'negreview', 'posreview',
            'answerscnt', 'otheragreemean', 'ratingmosthelpful', 'possentiment',
            'negsentiment', 'sentipolarity', 'reviewersrep', 'revpictotal',
            'prodpicstotal', 'ranking']

    # update 'training' table
    traindata.to_sql(tablename, con=connection, if_exists='append', index=False)


def trainSentiment(connection, cursor):
    '''
    create table named 'sentimenttraining' in product database.
    '''
    # Create training table
    sqlc = '''
        CREATE TABLE IF NOT EXISTS sentimenttraining (
        id TEXT NOT NULL,
        text TEXT,
        rating REAL,
        negsentiment INTEGER,
        possentiment INTEGER,
        ovsentiment INTEGER,
        FOREIGN KEY (id) REFERENCES prodpage(id)
            ON UPDATE CASCADE ON DELETE CASCADE
        )'''
    cursor.execute(sqlc)
    connection.commit()

def upTrainSentiment(connection, cursor, csvsentiment, tablename):
    # read csv file
    reviews = pd.read_csv(csvsentiment, skip_blank_lines=False)
    # exclude 'rowid'
    reviews = reviews[['id', 'text', 'rating', 'negsentiment', 'possentiment', 'ovsentiment']]
    # save it to sentimenttraining table in product.db
    reviews.to_sql(tablename, con=connection, if_exists='append', index=False)


