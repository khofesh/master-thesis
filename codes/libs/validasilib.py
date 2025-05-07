#!/usr/bin/env python3
'''
validasilib.py
This library contains functions needed for preprocessing.
- it could be used to import data from mongodb, extract
  _id and all the attributes of ordinaryReviewers and helpfulReviewers.
- it could be used to put all reviews to either csv or sqlite database.
- csv files columns:
  id | review | datereview | timereview | negsmiley | possmiley | neutsmiley
  | helpful | ordinary | profpic | prodpic | otheragree | rating | subsubcat
  | profpic2.
HOWTO:
    # connect to mongodb
    db = MongoClient().product
    prodpage = db['prodpage']
    # connect to sqlite
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    reviewertypes = ['ordinaryReviewers', 'helpfulReviewers']
    df1 = initialdf(prodpage, reviewertypes[0])
    df2 = initialdf(prodpage, reviewertypes[1])
    df, df1, df2 = mergedf(df1, df2)
    createcsvheader('test.csv')
    reviewtocsv(df, 'test.csv', prodpage)
    # you don't need to close mongodb connection

PERLU BUAT 2 TABLE DI SQLITE (product page reviews).
TABLE KEDUA, reviews, "id" MEMILIKI KONEKSI DENGAN "id" DI 
TABEL PERTAMA.
LU KALO MAU BUAT KOLOM BARU DI DATAFRAME, PAKE dataframe.insert(loc, column='test', value=0)
'''
# modules needed
import pymongo
from pymongo import MongoClient
import pprint
import pandas as pd
from bson.objectid import ObjectId
import subprocess
import sqlite3
from sqlite3 import Error
from time import sleep
import random
#opencv to compare images
import cv2 as cv
import numpy as np
import os
import csv
import shutil

def initialdf(collection, reviewer):
    '''
    * collection is the collection's name in mongodb
    database that you connected to.
    * reviewer is the kind of reviewer, either ordinary
    or helpful reviewer (ordinaryReviewers and helpfulReviewers).
    * ordinaryc/helpfulc --> 'c' means count
    '''
    reviewer = '$' + str(reviewer)
    df = []
    aggregate = collection.aggregate([{'$project': {'count': {'$size': reviewer}}}])
    for i in aggregate:
        df.append(i)
    df = pd.DataFrame(df)
    # select row which contains count bigger than
    # or equals 5
    # then sort dataframe's count from biggest to lowest
    if reviewer == '$ordinaryReviewers':
        df = df[df['count'] >= 5].sort_values('count', ascending=False)
        df.columns = ['id', 'ordinaryc']
        df['id'] = df['id'].astype(str)
    else:
        df = df.sort_values('count', ascending=False)
        df.columns = ['id', 'helpfulc']
        df['id'] = df['id'].astype(str)
    return df

def mergedf(df1, df2, how='inner', on='id'):
    '''
    merge 2 dataframe.
    default parameter:
    how='inner'
    on='id'
    '''
    df = pd.merge(df1, df2, how=how, on=on)
    df1 = None
    df2 = None
    return df, df1, df2

def helpfulReviewersDF(dfid, collection, reviewertypes='helpfulReviewers'):
    '''
    This function is written to simplify dataframe creation
    for helpfulReviewers ONLY.
    helpfulReviewers dataframe order:
    'commentGvnByReviewers', 'dateReviewers', 'imgReviewers',
    'negSmileyReviewers', 'neutSmileyReviewers', 'posSmileyReviewers',
    'ratingGvnByReviewers', 'timeReviewers'
    * attributes it doesn't have:
    'prodPicInReview' --> 'prodpic' = 'no'
    'reviewHelpful' --> otheragree = 0
    '''
    # prod contains product page attributes
    prod = collection.find_one({'_id': ObjectId(dfid)})
    # convert dict into dataframe
    helpfuldf = pd.DataFrame(prod[reviewertypes])
    # rename columns
    helpfuldf.columns = ['review', 'datereview', 'profpic', 'negsmiley',
            'neutsmiley', 'possmiley', 'rating', 'timereview']

    # convert 'review' string to lowercase
    helpfuldf['review'] = helpfuldf['review'].str.lower()
    # helpfulReviewers has no attribute 'prodpic', so 'no' for all
    helpfuldf.insert(8, column='prodpic', value=0)
    # helpfulReviewers has no attribute 'otheragree', so 0 for all
    # otheragree
    helpfuldf.insert(9, column='otheragree', value=0)
    # ordinary
    helpfuldf.insert(10, column='ordinary', value=0)
    # helpful
    helpfuldf.insert(11, column='helpful', value=1)
    # id
    helpfuldf.insert(12, column='id', value=dfid)
    # subsubcat
    helpfuldf.insert(13, column='subsubcat', value=prod['prodSubSubCat'])
    # profpic2
    helpfuldf.insert(14, column='profpic2', value=0)
    # sentiment
    helpfuldf.insert(15, column='sentiment', value=0)
    # reorder the columns
    helpfuldf = helpfuldf[['id', 'review', 'datereview', 'timereview',
        'negsmiley', 'possmiley', 'neutsmiley', 'helpful', 'ordinary', 
        'profpic', 'prodpic', 'otheragree', 'rating', 'subsubcat',
        'profpic2', 'sentiment']]

    return helpfuldf

def reviewtodf(dfid, collection, reviewertypes='ordinaryReviewers'):
    '''
    this function takes id in dataframe.
    The id is used to find data in mongodb database.
    dfid: id collection.
    howto: reviewdf = reviewtodf('5a988f12ae1f94099359dfea', prodpage)
    '''
    # prod contains product page attributes
    prod = collection.find_one({'_id': ObjectId(dfid)})
    ordinarydf = pd.DataFrame(prod[reviewertypes])

    # Show me 'id' and its columns' name.
    # FOR DEBUGGING PURPOSE ONLY. UNCOMMENT IF NOT NEEDED.
    # print("{0} and id: {1}".format(ordinarydf.columns, dfid))

    # Some mongodb documents have 'review' in dataframe columns.
    # EXCLUDE IT.
    ordinarydf = ordinarydf[['commentGvnByReviewersOrd', 'dateReviewersOrd', 'imgReviewersOrd',
       'negSmileyReviewersOrd', 'neutSmileyReviewersOrd',
       'posSmileyReviewersOrd', 'prodPicInReview', 'ratingGvnByReviewersOrd',
       'reviewOrdHelpful', 'timeReviewersOrd']]
    ordinarydf.columns = ['review', 'datereview', 'profpic', 'negsmiley',
            'neutsmiley', 'possmiley', 'prodpic', 'rating', 'otheragree', 'timereview']
    ordinarydf['ordinary'] = 1
    ordinarydf['helpful'] = 0
    ordinarydf['id'] = dfid
    ordinarydf['subsubcat'] = prod['prodSubSubCat']
    ordinarydf['profpic2'] = 0
    # Convert 'review' text to lower case.
    ordinarydf['review'] = ordinarydf['review'].str.lower()
    # Change 'yes'/'no' in prodpic column to 1/0 --> 1=True; 0=False
    ordinarydf['prodpic'] = ordinarydf['prodpic'].replace(
            ('yes', 'no'), (1,0))
    # Sentiment
    ordinarydf['sentiment'] = 0
    # reorder the columns
    ordinarydf = ordinarydf[['id', 'review', 'datereview', 'timereview',
        'negsmiley', 'possmiley', 'neutsmiley', 'helpful', 'ordinary', 
        'profpic', 'prodpic', 'otheragree', 'rating', 'subsubcat',
        'profpic2', 'sentiment']]

    # some products don't have 'helpfulReviewers'.
    # its count is 1 because it contains:
    # {'review': 'no review'}
    # The following code handles this problem
    if len(prod['helpfulReviewers']) == 1:
        try:
            if prod['helpfulReviewers'][0]['review'] == 'no review':
                #print('No Helpful Reviewers')
                return ordinarydf
        except KeyError:
            #print('There is One Helpful Reviewers')
            helpfuldf = helpfulReviewersDF(dfid, collection, 'helpfulReviewers')
            return pd.concat([ordinarydf, helpfuldf], ignore_index=True)
    else:
        helpfuldf = helpfulReviewersDF(dfid, collection, 'helpfulReviewers')
        return pd.concat([ordinarydf, helpfuldf], ignore_index=True)

def createcsvheader(filename):
    '''
    This function create a csv file that contains only
    the header/columns name.
    howto: createcsvheader('test.csv')
    '''
    columnsdf = pd.DataFrame(columns=['id', 'review', 'datereview', 
        'timereview', 'negsmiley', 'possmiley', 'neutsmiley', 
        'helpful', 'ordinary', 'profpic', 'prodpic', 'otheragree', 
        'rating'])
    columnsdf.to_csv(filename, mode='w', index=False)

def reviewtocsv(dataframe, filename, collection):
    '''
    dataframe : initialdf.
    This function append dataframe to a csv file that was created
    prior.
    howto: reviewtocsv(df, 'test.csv', prodpage)
    '''
    for i in range(len(dataframe['id'])):
        # create dataframe from dictionary of reviews
        reviewdf = reviewtodf(dataframe['id'][i], collection)
        # use append mode
        reviewdf.to_csv(filename, mode='a', header=False,
                index=False)

def downprofpic(dataframe, collection, directory, useragent):
    '''
    This function takes 'profpic' column and download
    the image using wget.
    useragent = 'something'
    howto: downprofpic(df, prodpage, './pictures/', useragent)
    '''
    for i in range(len(dataframe['id'])):
        reviewdf = reviewtodf(dataframe['id'][i], collection)
        # select column 'id' and 'profpic'
        reviewdf = reviewdf[['id', 'profpic']]
        for i in range(len(reviewdf)):
            wget = ['wget', '-cU']
            useragent = [useragent]
            url = [reviewdf['profpic'][i]]
            rename = ['-O']
            # Naming convention : starts from 1-end -> 1-id.jpg
            # (because sqlite rowid starts from 1) 
            # where id is mongodb _id.
            name = [str(i+1) + '-' + reviewdf['id'][i] + '.jpg']
            # Full wget command
            allcall = wget + useragent + url + rename + name 
            # Call wget command
            print('Download {0}'.format(name[0]))
            print(allcall)
            try:
                subprocess.call(allcall)
            except TypeError:
                print('Filename: {0} doesnt exist'.format(name[0]))
            # after the first loop, useragent becomes list.
            useragent = useragent[0]
            # Sleep randomly between 10-15 seconds
            # every multiple of 10.
            if i % 10 == 0:
                sleep(random.randint(5,10))
                print('Sleeping for appr. 10-20 seconds')
            else:
                pass

        # move all files with .jpg extension to specific folder.
        for filename in os.listdir():
            if filename.endswith('.jpg'):
                shutil.move(filename, directory)

def createProdPageTable(cursor, connection):
    '''
    This function creates table for product page in sqlite.
    'id' pada tabel reviews berhubungan dengan 'id' pada tabel ini.
    '''
    # sold/prodSold, seen/prodSeen, cashback, reputation, prodPic.
    # topads -> ubah menjadi 1/0
    # ranking is the result of implementing "ranking algorithm"
    # answerscnt: sum of questions answered.
    # questionscnt: sum of questions
    cursor.execute('''CREATE TABLE IF NOT EXISTS prodpage (
            id TEXT PRIMARY KEY,
            uri TEXT,
            prodname TEXT,
            prodpic BOOLEAN,
            price INTEGER,
            prodrating REAL,
            prodseen INTEGER,
            prodsold INTEGER,
            category TEXT,
            topads BOOLEAN,
            subcat TEXT,
            subsubcat TEXT,
            merchantname TEXT,
            reputation INTEGER,
            merchanttype TEXT,
            cashback REAL,
            rone INTEGER,
            rtwo INTEGER,
            rthree INTEGER,
            rfour INTEGER,
            rfive INTEGER,
            reviewcount INTEGER,
            discussionscnt INTEGER,
            questionscnt INTEGER,
            answerscnt INTEGER,
            helpfulreviewerscnt INTEGER,
            ranking INTEGER
            )''')
    connection.commit()

def createTable(cursor, connection):
    '''
    This function creates table in sqlite3 database.
    howto:
        conn = sqlite3.connect('product.db')
        c = conn.cursor()
        createTable(c, conn)
    id is related to id in prodpage table.
    ON UPDATE CASCADE -> when the parent key is modified, the values stored
        in each dependent child key are modified to match the new parent key values.
    ON DELETE CASCADE -> when the parent key is deleted, each row in the child table 
        that was associated with the deleted parent row is also deleted.
    '''
    cursor.execute('''CREATE TABLE IF NOT EXISTS reviews (
            id TEXT NOT NULL,
            review TEXT,
            datereview TEXT,
            timereview TEXT,
            negsmiley INTEGER,
            possmiley INTEGER, 
            neutsmiley INTEGER,
            helpful NUMERIC,
            ordinary NUMERIC,
            profpic TEXT,
            prodpic BOOLEAN,
            otheragree INTEGER,
            rating INTEGER,
            subsubcat TEXT,
            profpic2 BOOLEAN,
            sentiment INTEGER,
            FOREIGN KEY (id) REFERENCES prodpage(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )''')
    connection.commit()

def imagediff(avadir, profpicdir, csvfile):
    '''
    This function compares default avatars and other images, then 
    append the result to csv file.
    Return 0 if there is no differences, 1 otherwise.
    If there is no difference between default avatar and 
    the image that is tested, then reviewer whose picture is tested
    doesn't have profile picture.
    avadir : default avatar directory.
        Must be a string.
    profpicdir : directory path of all profile pictures.
        Must be a string.
    '''
    # Default avatar
    defaultava = os.listdir(avadir)
    # Full path for each default avatar .jpg file.
    defavapath = [avadir + i for i in defaultava]
    # Load the default avatar into vector.
    cvdefava = [cv.imread(i) for i in defavapath]

    # Profile pictures set directory
    profpicdir = profpicdir
    # Profile pictures name
    profpic = os.listdir(profpicdir)
    # Profile pictures fullpath
    profpicpath = [profpicdir + i for i in profpic]

    # Differences between each profile pictures and default 
    # avatar
    for i in profpicpath:
        diff = []
        cvprofpic = cv.imread(i)
        # Jika ada perbedaan, maka masukkan 1 pada diff
        for j in cvdefava:
            if np.any(cv.subtract(j, cvprofpic)):
                diff.append(1)
            else:
                diff.append(0)
        # If sum of diff equals 3, reviewer has profpic.
        # Get the sql id from picture's name
        sqlid = i.split('-')[1].split('.')[0]
        # Number in front of picture's name
        # 192-sqlid.jpg
        number = i.split('-')[0].replace('./profpics/', '')
        # append mode
        mode = 'a'
        # If the image is different from every default avatar,
        # sum([1, 1, 1]), then the image is the profile picture.
        if sum(diff) == len(defaultava):
            print('Reviewer has profile picture')
            zee = 1
            writetocsv(csvfile, mode, number, sqlid, zee)
        else:
            print('Reviewer has NO profile picture')
            zee = 0
            writetocsv(csvfile, mode, number, sqlid, zee)

def writetocsv(csvfile, mode, number, sqlid, zee):
    '''
    append variables to csv file.
    this function is written to simplify appending variables to
    csv file.
    written ONLY for imagediff function.
    zee : a number 1 or 0. 1 -> has profpic, 0 -> no profpic.
    howto: writetocsv(csvfile, mode='a', number, sqlid, zee)
    '''
    with open(csvfile, mode, newline='') as csvfile:
        filewritten = csv.writer(csvfile, delimiter=',')
        filewritten.writerow([number] + [sqlid] + [str(zee)])

def productdf(dfid, collection):
    '''
    Create product page dataframe from mongodb.
    '''
    prod = collection.find_one({'_id': ObjectId(dfid)})
    # Count helpful reviewers
    if len(prod['helpfulReviewers']) == 1:
        try:
            if prod['helpfulReviewers'][0]['review'] == 'no review':
                prod['helpfulreviewerscnt'] = 0
        except KeyError:
            prod['helpfulreviewerscnt'] = 1
    else:
        prod['helpfulreviewerscnt'] = len(prod['helpfulReviewers'])

    # Delete key 'helpfulReviewers' and 'ordinaryReviewers' from prod dictionary
    prod.pop('helpfulReviewers')
    prod.pop('ordinaryReviewers')
    # Convert ObjectId into string
    prod['_id'] = str(prod['_id'])
    # If prodDiscussions = [], meaning there's no discussion, then set it to 0,
    # otherwise do nothing.
    # example: prod['prodDiscussions'] -> [], it contains nothing/contains empty list.
    # empty list is False by default in python.
    if prod['prodDiscussions']:
        # Some database has prod['prodDiscussions'] -> [{'discussion': 'no discussion'}]
        try:
            prod['answerscnt'] = prod['prodDiscussions'][0]['answersCnt']
            prod['questionscnt'] = prod['prodDiscussions'][0]['questionsCnt']
        except KeyError:
            prod['answerscnt'] = 0
            prod['questionscnt'] = 0
    else:
        prod['prodDiscussions'] = 0
        prod['answerscnt'] = 0
        prod['questionscnt'] = 0
    
    # Convert prod dictionary into dictionary of list
    for i in prod.keys():
        prod[i] = [prod.get(i)]

    # Convert dictionary of list into dataframe
    df = pd.DataFrame(prod)
    # Change columns name
    df.columns = ['id', 'uri', 'topads', 'prodname', 'price', 'category', 'subcat',
            'subsubcat', 'merchantname', 'merchanttype', 'reputation', 'prodseen',
            'prodsold', 'prodpic', 'prodrating', 'reviewcount', 'discussionscnt', 
            'rfive', 'rfour', 'rthree', 'rtwo', 'rone', 'cashback', 'proddiscussions',
            'helpfulreviewerscnt', 'answerscnt', 'questionscnt']
    # Reorder the columns
    # proddiscussions di-exclude
    df = df[['id', 'uri', 'prodname', 'prodpic', 'price', 'prodrating', 'prodseen',
        'prodsold', 'category', 'topads', 'subcat', 'subsubcat', 'merchantname',
        'reputation', 'merchanttype', 'cashback', 'rone', 'rtwo', 'rthree', 'rfour',
        'rfive', 'reviewcount', 'discussionscnt', 'questionscnt', 'answerscnt', 
        'helpfulreviewerscnt']]
    df.insert(26, column='ranking', value=0)

    # Set product name to lowercase.
    df['prodname'] = df['prodname'].str.lower()

    # merchanttype: [Gold Merchant] --> 'gold' and [Official Store] --> 'official'
    # for testing uncomment the following:
    #print(df.loc[0, ])
    df['merchanttype'] = df['merchanttype'].str.replace('[Gold Merchant]', 'gold').replace('[Official Store]', 'official')

    # Cashback: 3% -> 0.03; "no cashback" -> 0; kemudian ubah ke float64
    # test
    df['cashback'] = df['cashback'].str.replace('%', '').replace('no cashback', '0').replace(',', '.').replace('\n', '0')
    df['cashback'] = df['cashback'].astype('float') / 100

    # prodsold
    def convText(text):
        '''
        convert 45rb into 45000 or 4,5rb into 4500
        '''
        try:
            test = text.replace('rb', '00')
            int(test.split('00')[0])
            test = test + '0'
            return test
        except ValueError:
            test = test.replace(',', '')
            return test

    try:
        df['prodsold'] = convText(df['prodsold'].values[0])
#        df['prodsold'] = df['prodsold'].str.replace('rb$', '00').str.replace(',', '')
        df['prodsold'] = df['prodsold'].astype('int')
    except ValueError:
        print('prodsold ValueError; id = {0}'.format(dfid))
        df['prodsold'] = df['prodsold'].str.strip().str.replace('rb$', '00').str.replace(',', '')
        df['prodsold'] = 0
        df['prodsold'] = df['prodsold'].astype('int')

    # prodseen
    try:
        df['prodseen'] = convText(df['prodseen'].values[0])
#        df['prodseen'] = df['prodseen'].str.replace('rb$', '00').str.replace(',', '')
        df['prodseen'] = df['prodseen'].astype('int')
    except ValueError:
        print('prodseen ValueError; id = {0}'.format(dfid))
        df['prodseen'] = df['prodseen'].str.strip().str.replace('rb$', '00').str.replace(',', '')
        df['prodseen'] = 0
        df['prodseen'] = df['prodseen'].astype('int')

    # prodrating
    try:
        df['prodrating'] = df['prodrating'].astype('float')
    except ValueError:
        print('prodrating ValueError; id = {0}'.format(dfid))
        df['prodrating'] = 0
        df['prodrating'] = df['prodrating'].astype('float')

    # price
    df['price'] = df['price'].str.replace('Rp ', '')
    df['price'] = df['price'].astype('int')

    # topads; yes -> 1; no -> 0
    df['topads'] = df['topads'].replace(('yes', 'no'), (1,0))
    #df['topads'] = df['topads'].astype('bool')

    # prodpic
    df['prodpic'] = df['prodpic'].replace(('yes', 'no'), (1,0))

    # rone-rfive
    df['rone'] = df['rone'].astype('int')
    df['rtwo'] = df['rtwo'].astype('int')
    df['rthree'] = df['rthree'].astype('int')
    df['rfour'] = df['rfour'].astype('int')
    df['rfive'] = df['rfive'].astype('int')

    # reviewcount
    df['reviewcount'] = df['reviewcount'].str.replace('rb$', '00').str.replace(',', '')
    df['reviewcount'] = df['reviewcount'].astype('int')

    # discussioncnt
    df['discussionscnt'] = df['discussionscnt'].str.replace('rb$', '00').str.replace(',', '')
    df['discussionscnt'] = df['discussionscnt'].astype('int')

    # reputation
    # some record has df['reputation'] -> 'Official Store'
    try:
        df['reputation'] = df['reputation'].astype('int')
    except ValueError:
        print('This id: {0} has \'reputation\' value = \'Official Store\''.format(dfid))
        df['reputation'] = 0

    return df

def prodDfToSqlite(initialdf, tablename, collection, conn):
    '''
    This function append product page dataframe into sqlite file.
    '''
    for i in range(len(initialdf['id'])):
        # create product page dataframe
        proddf = productdf(initialdf['id'][i], collection)
        # convert df to sqlite
        proddf.to_sql(tablename, con=conn, 
                if_exists='append', index=False)

def dftosqlite(dataframe, tablename, collection, conn):
    '''
    This function append dataframe to a sqlite db that was created
    prior.
    howto: reviewtocsv(df, 'test.csv', prodpage)
    '''
    for i in range(len(dataframe['id'])):
        # create dataframe from dictionary of reviews
        reviewdf = reviewtodf(dataframe['id'][i], collection)
        # convert df to sqlite. "flavor" is deprecated
        reviewdf.to_sql(tablename, con=conn, if_exists='append', index=False)

def correctError(variable, mongoid, dbcursor, correctvalue, dbconnection):
    '''
    This function will ask user which record (mongodb id) has a wrong value 
    in its variable.
    The user then asked to provide the actual value of the variable.
    '''
    if variable == 'reputation':
        sql = '''UPDATE prodpage SET reputation = ? 
                WHERE id = ?'''
        dbcursor.execute(sql, (correctvalue, mongoid))
    elif variable == 'prodseen':
        sql = '''UPDATE prodpage SET prodseen = ? 
                WHERE id = ?'''
        dbcursor.execute(sql, (correctvalue, mongoid))
    elif variable == 'prodsold':
        sql = '''UPDATE prodpage SET prodsold = ? 
                WHERE id = ?'''
        dbcursor.execute(sql, (correctvalue, mongoid))
    elif variable == 'prodrating':
        sql = '''UPDATE prodpage SET prodrating = ? 
                WHERE id = ?'''
        dbcursor.execute(sql, (correctvalue, mongoid))
    else:
        print('Not supported')

    dbconnection.commit()

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        conn.close()


