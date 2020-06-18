#!/usr/bin/env python3
'''
Preprocessing
pdownpics.py
reviews - download - pictures
download reviewers' profile picture
'''

from lib.reviewslib import *

# connect to mongodb
db = MongoClient().product
prodpage = db['prodpage']

# create initial dataframe
reviewertypes = ['ordinaryReviewers', 'helpfulReviewers']
df1 = initialdf(prodpage, reviewertypes[0])
df2 = initialdf(prodpage, reviewertypes[1])
df, df1, df2 = mergedf(df1, df2)
# Karena mati lampu
df = df.loc[0:36,]
df.index = range(0,len(df))

# download profile pictures of reviewers
useragent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"
folder = './profpics/'
downprofpic(df, prodpage, folder, useragent)
# you don't need to close mongodb connection
