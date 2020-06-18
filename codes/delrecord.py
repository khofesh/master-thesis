'''
Delete row in sqlite database if it's not in 'Elektronik ' category
'''

# library
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# delete rows
sqlcommand = 'DELETE FROM prodpage WHERE category != "Elektronik "'
c.execute(sqlcommand)
conn.commit()
conn.close()
