'''
Preprocessing.
This script update prodpic in reviews table where 
helpful == 1.
'''

import pandas as pd
import sqlite3
from tqdm import tqdm


def updateProdpic(sqlid, hrcount, picreview1, picreview2, picreview3, sqlconn, cursor):
    '''
    sqlid = mongodb id
    hrcount = helpfulreviewerscnt (3, 2, 1)
    sqlconn = sqlite3 connection
    cursor = sqlite3 cursor
    dcommand<number> = default command number 1-?
    picreview1 - picreview3 = integer 0 or 1
    '''
    # to query rowid and id in table reviews
    dcommand1 = 'SELECT rowid, id, prodpic FROM reviews WHERE id = '
    dcommand2 = ' AND helpful == 1'
    # to update prodpic value in table reviews
    dcommand3 = 'UPDATE reviews SET prodpic = '
    dcommand4 = ' WHERE id = '
    dcommand5 = ' AND rowid = '

    # sqlid
    sqlid = '"' + sqlid + '"'

    # full sql command to query
    fquery = dcommand1 + sqlid + dcommand2
    cursor.execute(fquery)
    sqlconn.commit()
    # query result
    ''' example: 
    [(176926, '5aa67a6935d6d31e3fe26768', 0),
    (176927, '5aa67a6935d6d31e3fe26768', 0)]
    qresult[0][0] -> 176926 (rowid)
    qresult[1][0] -> 176927 (rowid)
    '''
    qresult = cursor.fetchall()

    if hrcount == 3:
        fupdate1 = dcommand3 + picreview1 + dcommand4 + sqlid + dcommand5 + str(qresult[0][0])
        cursor.execute(fupdate1)
        sqlconn.commit()
        fupdate2 = dcommand3 + picreview2 + dcommand4 + sqlid + dcommand5 + str(qresult[1][0])
        cursor.execute(fupdate2)
        sqlconn.commit()
        fupdate3 = dcommand3 + picreview3 + dcommand4 + sqlid + dcommand5 + str(qresult[2][0])
        cursor.execute(fupdate3)
        sqlconn.commit()
    elif hrcount == 2:
        fupdate1 = dcommand3 + picreview1 + dcommand4 + sqlid + dcommand5 + str(qresult[0][0])
        cursor.execute(fupdate1)
        sqlconn.commit()
        fupdate2 = dcommand3 + picreview2 + dcommand4 + sqlid + dcommand5 + str(qresult[1][0])
        cursor.execute(fupdate2)
        sqlconn.commit()
    elif hrcount == 1:
        fupdate1 = dcommand3 + picreview1 + dcommand4 + sqlid + dcommand5 + str(qresult[0][0])
        cursor.execute(fupdate1)
        sqlconn.commit()
    else:
        print('Tokopedia product page has only 3, 2, or 1 helpful reviewers')
        print('hrcount: {0}; id: {1}'.format(hrcount, sqlid))


# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# read csv file
# hrprodpic = helpful reviewers product picture
# hasil scraping menggunakan crawler hrhaspics
hrprodpic = pd.read_csv('./csvfiles/helpfulReviewersProdPics.csv')

# remove duplicate uri
hrprodpic = hrprodpic[~hrprodpic.duplicated(subset='uri', keep='first')]
# change index
hrprodpic.index = range(0, len(hrprodpic))

defaultcommand = 'SELECT id, helpfulreviewerscnt FROM prodpage WHERE uri = '

for i in tqdm(range(len(hrprodpic))):
    uri = '"' + hrprodpic.loc[i, ]['uri'] + '"'
    fullcommand = defaultcommand + uri
    # execute sql command
    c.execute(fullcommand)
    conn.commit()

    # result (a list of tupple) -> [('id', helpfulreviewerscnt)]
    retrieved = c.fetchall()
    # sum of helpful reviewers
    hrcount = retrieved[0][1]
    # helpful reviews contains product pictures (0/1)
    picreview1 = str(hrprodpic.loc[i, ]['picreview1'])
    picreview2 = str(hrprodpic.loc[i, ]['picreview2'])
    picreview3 = str(hrprodpic.loc[i, ]['picreview3'])
    # sqlid
    sqlid = retrieved[0][0]

    # update 'prodpic' value in table 'reviews'
    updateProdpic(sqlid, hrcount, picreview1, picreview2, picreview3, conn, c)

conn.close()
