'''
Preprocessing
Correct unexpected value for variable:
prodseen, reputation, prodsold, and prodrating.
'''

# import modules from reviewslib
from lib.reviewslib import *

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# make connection to mongodb
# db name:product :: collection name:prodpage
db = MongoClient().product
prodpage = db.prodpage

# Read csv file that contains id and variable name that contains error
df = pd.read_csv('./csvfiles/ValueError.csv')

# Fix error
for i in range(len(df)):
    print('No. {0} | id: {1} | var: {2}'.format(i, df['id'][i], df['variable'][i]))
    produri = prodpage.find_one({'_id': ObjectId(df['id'][i])})['uri']
    print('uri: {0}'.format(produri))
    realvalue = input('Input real value: ')
    if df['variable'][i] == 'prodrating':
        realvalue = float(realvalue)
    else:
        realvalue = int(realvalue)
    correctError(df['variable'][i], df['id'][i], c, realvalue, conn)
    # Not needed, it was implemented in correctError function.
    # conn.commit()


conn.close()
