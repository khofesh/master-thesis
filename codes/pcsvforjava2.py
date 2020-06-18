'''
Preprocessing.
Generate csv file that contains rowid, id, review untuk 
text preprocessing.
'''

import pandas as pd
import sqlite3
import re

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# query
sqlcommand = 'SELECT review FROM reviews'
c.execute(sqlcommand)
conn.commit()

# from book "python machine learning"
def preprocessor(text):
    # ð<9f><98><8a>
    # recognized by variable text below -> <9f><98><8a>
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=) (?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

# turn it into dataframe
text = c.fetchall()
text = pd.DataFrame(text)
# change column's name
text.columns = ['review']
# do some cleaning.
text['review'] = text['review'].astype('str')
text['review'] = text['review'].str.strip()
text['review'] = text['review'].str.replace('\n', ' ')
text['review'] = text['review'].apply(preprocessor)
#text['review'] = text['review'].str.replace('ð', '')

# import it into csv file
filename = './csvfiles/input_reviewtext.csv'
text.to_csv(filename, header=False, index=False)
