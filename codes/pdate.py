# coding: utf-8
'''
Preprocessing.
Update datereview formatting in sqlite.
'''

# import modules from reviewslib
from lib.reviewslib import *
from tqdm import tqdm

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

rows = pd.read_table('./csvfile/sqlCommand.txt', names=['command'])

for i in tqdm(range(len(rows))):
    '''
    fd = open('sqlCommand.txt', 'a')
    sql = "UPDATE reviews SET datereview = " + '"' + rows['date'][i] + '"' + " WHERE id = " + '"' + rows['id'][i] + '"' + " AND rowid =" + str(rows['rowid'][i]) + ";" + "\n"
    fd.write(sql)
    fd.close()
    '''
    c.execute(rows['command'][i])
    conn.commit()
    #command = "UPDATE reviews SET datereview = ? WHERE id = ? AND rowid = ?"
    #print('update {0} | {1} | {2}'.format(rows['date'][i], rows['id'][i], rows['rowid'][i]))
    #c.execute(command, (rows['date'][i], rows['id'][i], rows['rowid'][i]))
    #conn.commit()

conn.close()
