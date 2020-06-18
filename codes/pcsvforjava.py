'''
Preprocessing.
Generate csv file that contains rowid, id, review untuk 
text preprocessing.
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
sqlcommand = 'SELECT rowid, id, review FROM reviews'
c.execute(sqlcommand)
conn.commit()

# turn it into dataframe
text = c.fetchall()
text = pd.DataFrame(text)
# change column's name
text.columns = ['rowid', 'id', 'review']
# do some cleaning.
text['review'] = text['review'].astype('str')
text['rowid'] = text['rowid'].astype('int')
text['review'] = text['review'].str.strip()
text['review'] = text['review'].str.replace('\n', ' ')

# import it into csv file
filename = './csvfiles/reviewTextTobeProcessed.csv'
text.to_csv(filename, header=False, index=False)
