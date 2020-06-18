'''
pprodhasur.py
Preprocessing - Product Page Has Useful Reviewers.
This script query product.db , then create a csv file 
    that contains id, uri, helpfulreviewerscnt.
'''

import sqlite3
import pandas as pd
from tqdm import tqdm

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# select product page that has helpfulreviewerscnt > 0
sqlcommand = '''SELECT id, uri, helpfulreviewerscnt FROM prodpage 
                WHERE helpfulreviewerscnt != 0'''
c.execute(sqlcommand)
conn.commit()

data = c.fetchall()
data = pd.DataFrame(data)
data.columns = ['id', 'uri', 'helpfulreviewerscnt']
data.to_csv('./csvfiles/productHasHelpfulReviewers.csv', index=False)
