'''
Preprocess product name and subsubcategory.
'''

# library
import pandas as pd
import numpy as np
import sqlite3

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

# query
sqlcommand = 'SELECT id, prodname, subcat, subsubcat FROM prodpage WHERE category == "Elektronik "'
c.execute(sqlcommand)
conn.commit()

product = c.fetchall()
product = pd.DataFrame(product)
product.columns = ['id', 'prodname', 'subcat', 'subsubcat']
product['subcat'] = product['subcat'].astype('category')
product['subsubcat'] = product['subsubcat'].astype('category')

# save product['prodname'] to csv file, prodname.csv
product['prodname'].to_csv('./csvfiles/input_prodname.csv', index=False)

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(y='subcat', data=product, palette="Greens_d")
sns.countplot(y='subsubcat', data=product, palette="Greens_d")
###############################################################

# Preprocess output_prodname.csv
names = pd.read_csv('./csvfiles/output_prodname.csv', skip_blank_lines=False, header=None)
names.columns = ['prodname']
names[names['prodname'].isnull()]
names['prodname'] = names['prodname'].replace(np.nan, '', regex=True)

# Stemming
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    tokenized = [porter.stem(word) for word in text.split()]
    # join strings inside list
    tokenized = ' '.join(tokenized)
    return tokenized
# tokenize review['text']
names['prodname'] = names['prodname'].apply(tokenizer_porter)

# stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
def stopwordsEnglish(text):
    nostopwords = [w for w in text.split() if w not in stop]
    nostopwords = ' '.join(nostopwords)
    return nostopwords
# delete stopwords
names['prodname'] = names['prodname'].apply(stopwordsEnglish)



conn.close()
