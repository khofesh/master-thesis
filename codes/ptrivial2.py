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

# read csvfile
csvinput = './csvfiles/merchanttype.csv'
prodpage = pd.read_csv(csvinput)

sqlc = 'UPDATE prodpage SET '
sqlc1 = 'merchanttype = '
sqlc2 = 'WHERE merchantname = '

# Manually update merchanttype with merchantname:
# Kamera cctv "murah",  BALBOA "Grosir Jaya", " 1st Store ", Vin"z Shop Fashion
# then delete the row in csvinput

print('Updating sql column \'merchanttype\'')
for i in tqdm(range(len(prodpage))):
    merchantname = prodpage.loc[i, 'merchantname']
    merchanttype = prodpage.loc[i, 'merchanttype']
    sqlall = sqlc + sqlc1 + '"' + merchanttype + '"' + ' ' + sqlc2 + '"' + merchantname + '"'
    print(sqlall)
    c.execute(sqlall)
    conn.commit()
