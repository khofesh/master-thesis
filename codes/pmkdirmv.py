'''
make directories inside profpics folder 
    named after mongodb '_id' or sqlite3 'id'
'''

# library
import os
import sqlite3
import glob
import shutil
from tqdm import tqdm

# sqlite connection
conn = sqlite3.connect('product.db')
c = conn.cursor()

sqlcommand = 'SELECT id FROM prodpage'
c.execute(sqlcommand)
conn.commit()
# mongodb id
mongoid = c.fetchall()

# directory
profpics = os.getcwd() + '/profpics/'

for i in tqdm(range(len(mongoid))):
    dirname = mongoid[i][0]
    fullpath = profpics + dirname
    if not os.path.exists(fullpath):
        # create directory
        os.makedirs(fullpath)
    # to match every .jpg file that has current mongoid
    contains = '*' + mongoid[i][0] + '*.jpg'
    # find .jpg file that match current mongodb id
    for picfile in glob.glob(profpics + contains):
        print(picfile)
        # move those files to 'fullpath'
        shutil.move(picfile, fullpath)
