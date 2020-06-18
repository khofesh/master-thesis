#!/usr/bin/env python3
'''
mongotosqlite.py
This script convert mongodb database into sqlite.
create sqlite3 files for all reviews.
mongodb '_id' should be the "id"
'''
# import modules from reviewslib
from lib.reviewslib import *

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# make connection to mongodb
# db name:product :: collection name:prodpage
db = MongoClient().product
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
