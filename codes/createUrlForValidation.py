'''
create urlValidasi.csv and store it inside scraper 'validasi'
'''

import pandas as pd
import sqlite3

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# query
sqlc = 'SELECT id, uri, actualrevcount, salescluster FROM prodpage WHERE actualrevcount >= '
revcount = 30
sqlcall = sqlc + str(revcount)
c.execute(sqlcall)
conn.commit()

# turn it into dataframe
prod = c.fetchall()
prod = pd.DataFrame(prod)
prod.columns = ['id', 'uri', 'actualrevcount', 'salescluster']

# query 2
sqlc2 = 'SELECT id, uri, actualrevcount, salescluster FROM prodpage WHERE salescluster == 2.0 AND actualrevcount >= '
revcount2 = 25
sqlcall2 = sqlc2 + str(revcount2)
c.execute(sqlcall2)
conn.commit()

# dataframe 2
prod2 = c.fetchall()
prod2 = pd.DataFrame(prod2)
prod2.columns = ['id', 'uri', 'actualrevcount', 'salescluster']

# join or merge
result = pd.concat([prod, prod2])

# folder path
destpath = './crawler/validasi/validasi/spiders/urlValidasi.csv'
result.to_csv(destpath, header=False, index=False)
