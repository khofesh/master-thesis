#!/usr/bin/env python3
'''
validasisqlite.py
this script convert mongodb database named 'validasi' into sqlite 

Testing:
    import pymongo
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    client = MongoClient()
    db = client['validasi']
    collection = db['prodpage']
    prod = collection.find_one({'_id': ObjectId('5b32126eb8a9f70142f6a766')})
    if len(prod['helpfulReviewers']) == 1:
        try:
            if prod['helpfulReviewers'][0]['review'] == 'no review':
                prod['helpfulreviewerscnt'] = 0
        except KeyError:
            prod['helpfulreviewerscnt'] = 1
    else:
        prod['helpfulreviewerscnt'] = len(prod['helpfulReviewers'])
    prod.pop('helpfulReviewers')
    prod.pop('ordinaryReviewers')
    prod['_id'] = str(prod['_id'])
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

'''
# import modules from reviewslib
from lib.validasilib import *

# make connection to sqlite db
conn = sqlite3.connect('validasi.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# make connection to mongodb
# db name:validasi :: collection name:prodpage
db = MongoClient().validasi
prodpage = db.prodpage

reviewertypes = ['ordinaryReviewers', 'helpfulReviewers']
df1 = initialdf(prodpage, reviewertypes[0])
df2 = initialdf(prodpage, reviewertypes[1])
df, df1, df2 = mergedf(df1, df2)

# create product page table
createProdPageTable(c, conn)

# create review table
createTable(c, conn)

# append dataframe to sqlite
# prodDfToSqlite(initialdf, tablename, collection, conn)
prodDfToSqlite(df, 'prodpage', prodpage, conn)

# dftosqlite(dataframe, tablename, collection, conn)
dftosqlite(df, 'reviews', prodpage, conn)

# close connection
conn.close()
# you don't need to close mongodb connection
