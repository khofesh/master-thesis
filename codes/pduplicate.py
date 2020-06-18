'''
Preprocessing.
Delete record which has duplicate uri in prodpage table.
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

# select id and uri
sqlcommand = "SELECT id, uri FROM prodpage"
c.execute(sqlcommand)
conn.commit()

data = c.fetchall()
data = pd.DataFrame(data)
data.columns = ['id', 'uri']

# select record that has duplicate data
notduplicate = data[~data.duplicated(subset='uri', keep='first')]
duplicate = data[data.duplicated(subset='uri', keep='first')]

# set new index for notduplicate and duplicate
notduplicate.index = range(0,len(notduplicate))
duplicate.index = range(0, len(duplicate))

# compare reviews count between links found in duplicate and notduplicate
for i in tqdm(range(len(duplicate))):
    defaultcommand = "SELECT datereview FROM reviews WHERE id = "
    dupID = '"' + duplicate.loc[i, ]['id'] + '"'
    notdupID = '"' + notduplicate[notduplicate['uri'] == duplicate.loc[i,]['uri']]['id'].values[0] + '"'

    # select anything, it doesnt matter. What matters is its count.
    # for duplicated data
    sqlcommand = defaultcommand + dupID
    c.execute(sqlcommand)
    conn.commit()
    reviewCntDup = len(c.fetchall())
    # not duplicate
    sqlcommand = defaultcommand + notdupID
    c.execute(sqlcommand)
    conn.commit()
    reviewCntNotDup = len(c.fetchall())

    if reviewCntDup > reviewCntNotDup:
        # Delete row in sql
        print('MORE; Delete row where id: {0}'.format(dupID))
        delcommand = 'DELETE FROM prodpage WHERE id = '
        delcommand = delcommand + notdupID
        c.execute(delcommand)
        conn.commit()
    elif reviewCntDup < reviewCntNotDup:
        # Delete row in sql
        print('LESS; Delete row where id: {0}'.format(dupID))
        delcommand = 'DELETE FROM prodpage WHERE id = '
        delcommand = delcommand + dupID
        c.execute(delcommand)
        conn.commit()
    elif reviewCntDup == reviewCntNotDup:
        # Delete row in sql
        print('SAME; Delete row where id: {0}'.format(dupID))
        delcommand = 'DELETE FROM prodpage WHERE id = '
        delcommand = delcommand + dupID
        c.execute(delcommand)
        conn.commit()

conn.close()
# Too much repetition???
# YES, I'm lazy to write function.
