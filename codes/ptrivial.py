import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()


sqlc = 'SELECT id, uri, merchantname, merchanttype FROM prodpage'
c.execute(sqlc)
conn.commit()
prodpage = c.fetchall()
prodpage = pd.DataFrame(prodpage)
prodpage.columns = ['id', 'uri', 'merchantname', 'merchanttype']

# delete duplicate
prodpage = prodpage[~prodpage['merchantname'].duplicated(keep='first')]

# write to csv file
csvouput = './csvfiles/merchanttype.csv'
prodpage.to_csv(csvouput, index=False)
