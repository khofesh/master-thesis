'''
Sentiment analysis
'''

# library
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
from lib.sentistrength.sentistrength_id import *

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

'''
review = pd.read_csv('./csvfiles/output_englishstem.csv', skip_blank_lines=False, low_memory=False)
review['text'] = review['text'].astype('str')

# wordcloud
from wordcloud import WordCloud
# WordCloud().generate expect string not pd.Series
semuatext = []
for i in range(len(review['text'])):
    for j in review['text'][i].split():
        semuatext.append(j)

semuatext = sorted(semuatext)
semuatext = ' '.join(alltext)
wordcloud = WordCloud().generate(alltext)
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# versi lain
from collections import Counter
semuatext = []
for i in range(len(review['text'])):
    for j in review.loc[i, 'text'].split():
        semuatext.append(j)
counts = Counter(semuatext)
counts.get('terima')
counts.get('barang')
counts.get('kasih')
wordcloud = WordCloud(background_color='white', max_words=200).generate_from_frequencies(counts)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
###############################################################
'''

def englishStem(csvfile, conn, cursor, output):
    '''
    csvfile: path where review text preprocessing (InaNLP) output resides
    conn: sqlite3 connection
    cursor: sqlite3 cursor
    output: path where you want to save your output to
    '''
    # enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    # query
    sqlcommand = 'SELECT rowid, id, rating FROM reviews'
    cursor.execute(sqlcommand)
    conn.commit()
    
    rating = cursor.fetchall()
    rating = pd.DataFrame(rating)
    rating.columns = ['rowid', 'id', 'rating']
    
    # out1.csv
    # review text that have been preprocessed
    review = pd.read_csv(csvfile, skip_blank_lines=False, header=None)
    review.columns = ['text']
    
    # append review to rating
    review = rating.join(review)
    review.columns = ['rowid', 'id', 'rating', 'text']
    review = review[['rowid', 'id', 'text', 'rating']]
    
    # remove every number in review['text']
    review['text'] = review['text'].str.replace('\d+', '')
    ###############################################################

    # DEALING WITH NaN and none
    
    # review[review['rating'].isnull()]
    # Out[53]: 
    #          rowid                        id  text  rating
    # 70916    74203  5a8daf060a9829061fafcc62  none     NaN
    # 177577  185675  5a8db2500a9829061fafcc63  none     NaN
    review['rating'] = review['rating'].astype('float64')
    review.loc[70916, 'rating'] = 0
    review.loc[177577, 'rating'] = 0
    
    # review.info()
    # review[pd.isna(review['text'])]
    # review[review['text'] == 'none']
    review.loc[70916, 'text'] = np.nan
    review.loc[177577, 'text'] = np.nan
    # replace nan with empty string
    review['text'] = review['text'].replace(np.nan, '', regex=True)
    
    # some words in review['text'] are in english
    # Stemming
    from nltk.stem.snowball import EnglishStemmer
    snowball = EnglishStemmer()
    def tokenizer_snowball(text):
        tokenized = [snowball.stem(word) for word in text.split()]
        # join strings inside list
        tokenized = ' '.join(tokenized)
        return tokenized
    # tokenize review['text']
    review['text'] = review['text'].apply(tokenizer_snowball)
    
    # stopwords
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    def stopwordsEnglish(text):
        nostopwords = [w for w in text.split() if w not in stop]
        nostopwords = ' '.join(nostopwords)
        return nostopwords
    # delete stopwords
    review['text'] = review['text'].apply(stopwordsEnglish)

    # save it to csv file
    review.to_csv(output, index=False)

def classSentiment(inputfile, outputfile):
    '''
    classify sentiment strength
    '''
    
    config = dict()
    config["negation"] = True
    config["booster"]  = True
    config["ungkapan"]  = True
    config["consecutive"]  = True
    config["repeated"]  = True
    config["emoticon"]  = True
    config["question"]  = True
    config["exclamation"]  = True
    config["punctuation"]  = True
    senti = sentistrength(config)

    review = pd.read_csv(inputfile, skip_blank_lines=False)
    review['text'] = review['text'].replace(np.nan, '', regex=True)
    review['negsentiment'] = 0
    review['possentiment'] = 0
    # overal sentiment
    review['ovsentiment'] = 0
    
    for i in tqdm(range(len(review))):
        allsentiment = senti.main(review.loc[i, 'text'])
        review.loc[i, 'negsentiment'] = allsentiment['max_negative']
        review.loc[i, 'possentiment'] = allsentiment['max_positive']
        if allsentiment['kelas'] == 'positive':
            review.loc[i, 'ovsentiment'] = 1
        elif allsentiment['kelas'] == 'negative':
            review.loc[i, 'ovsentiment'] = -1
        elif allsentiment['kelas'] == 'neutral':
            review.loc[i, 'ovsentiment'] = 0
        else:
            print('What are you trying to do?')

    review.to_csv(outputfile, index=False)

def addCol(connection, cursor):
    sqlc1 = 'ALTER TABLE reviews RENAME TO reviewsOld'
    cursor.execute(sqlc1)
    connection.commit()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reviews (
            id TEXT NOT NULL,
            review TEXT,
            datereview TEXT,
            timereview TEXT,
            negsmiley INTEGER,
            possmiley INTEGER,
            neutsmiley INTEGER,
            helpful NUMERIC,
            ordinary NUMERIC,
            profpic TEXT,
            prodpic BOOLEAN,
            otheragree INTEGER,
            rating INTEGER,
            subsubcat TEXT,
            profpic2 BOOLEAN,
            negsentiment INTEGER,
            possentiment INTEGER,
            ovsentiment INTEGER,
            FOREIGN KEY (id) REFERENCES prodpage(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )''')
    connection.commit()
    sqlc2 = '''
        INSERT INTO reviews (id, review, datereview, timereview, negsmiley, possmiley, neutsmiley, helpful, ordinary, profpic, prodpic, otheragree, rating, subsubcat, profpic2, negsentiment, possentiment, ovsentiment) 
        SELECT id, review, datereview, timereview, negsmiley, possmiley, neutsmiley, helpful, ordinary, profpic, prodpic, otheragree, rating, subsubcat, profpic2, negsentiment, possentiment, ovsentiment FROM reviewsOld
        '''
    cursor.execute(sqlc2)
    connection.commit()
    sqlc3 = 'DROP TABLE reviewsOld'
    cursor.execute(sqlc3)
    connection.commit()

def upReviewsTable(csvinput, connection, cursor):
    # fix rowid
    fixc = 'SELECT rowid, id FROM reviews'
    cursor.execute(fixc)
    connection.commit()
    fixrowid = cursor.fetchall()
    fixrowid = pd.DataFrame(fixrowid)
    fixrowid.columns = ['rowid', 'id']

    sqlc1 = 'UPDATE reviews SET '
    sqlc2 = 'negsentiment = '
    sqlc3 = 'possentiment = '
    sqlc4 = 'ovsentiment = '
    sqlc5 = 'WHERE id = '
    sqlc6 = 'AND rowid = '

    review = pd.read_csv(csvinput, skip_blank_lines=False)
    # change 'review' dataframe 'rowid'
    review['rowid'] = fixrowid['rowid']

    for i in tqdm(range(len(review))):
        negsenti = str(review.loc[i, 'negsentiment']) + ', '
        possenti = str(review.loc[i, 'possentiment']) + ', '
        ovsenti = str(review.loc[i, 'ovsentiment']) + ' '
        mongoid = '"' + review.loc[i, 'id'] + '" '
        rowid = str(review.loc[i, 'rowid'])
        sqlall = sqlc1 + sqlc2 + negsenti + sqlc3 + possenti + \
                sqlc4 + ovsenti + sqlc5 + mongoid + sqlc6 + rowid
        cursor.execute(sqlall)
        connection.commit()

    # convert review dataframe into csv file
    review.to_csv('./csvfiles/output_sentiment_fixedrowid.csv', index=False)

def addColProdpage(connection, cursor):
    '''
    add columns (possentiment, negsentiment, sentipolarity) to prodpage table
    '''
    sqlc1 = 'ALTER TABLE prodpage ADD COLUMN possentiment INTEGER'
    cursor.execute(sqlc1)
    connection.commit()
    sqlc2 = 'ALTER TABLE prodpage ADD COLUMN negsentiment INTEGER'
    cursor.execute(sqlc2)
    connection.commit()
    sqlc3 = 'ALTER TABLE prodpage ADD COLUMN sentipolarity INTEGER'
    cursor.execute(sqlc3)
    connection.commit()

def upProdpageTable(csvinput, connection, cursor):
    '''
    update column (possentiment, negsentiment, sentipolarity) in prodpage table
    '''
    sqlc1 = 'SELECT id, possentiment, negsentiment, sentipolarity FROM prodpage'
    cursor.execute(sqlc1)
    connection.commit()
    # fetch all query result
    product = cursor.fetchall()
    product = pd.DataFrame(product)
    product.columns = ['id', 'possentiment', 'negsentiment', 'sentipolarity']
    product['id'] = product['id'].astype('str')

    # read csvfile
    reviews = pd.read_csv(csvinput, skip_blank_lines=False)
    # change id type to category
    reviews['id'] = reviews['id'].astype('category')
    # groupby id
    # for average degree of sentiment strength
    reviewsMean = reviews.groupby(['id']).mean().round()
    reviewsMean['negsentiment'] = reviewsMean['negsentiment'].astype('int')
    reviewsMean['possentiment'] = reviewsMean['possentiment'].astype('int')
    reviewsMean['ovsentiment'] = reviewsMean['ovsentiment'].astype('int')
    # for sentiment polarity
    reviewsSum = reviews.groupby(['id']).sum()
    reviewsSum['negsentiment'] = reviewsSum['negsentiment'].astype('int')
    reviewsSum['possentiment'] = reviewsSum['possentiment'].astype('int')
    reviewsSum['ovsentiment'] = reviewsSum['ovsentiment'].astype('int')

    print('Assign product dataframe\'s sentiment strength and polarity')
    for i in tqdm(range(len(product))):
        mongoid = product.loc[i, 'id']
        product.loc[i, 'possentiment'] = reviewsMean[reviewsMean.index == mongoid]['possentiment'].values[0]
        product.loc[i, 'negsentiment'] = reviewsMean[reviewsMean.index == mongoid]['negsentiment'].values[0]
        # overall sentiment (sum)
        ovsentiment = reviewsSum[reviewsSum.index == mongoid]['ovsentiment'].values[0]
        if ovsentiment > 0:
            product.loc[i, 'sentipolarity'] = 1
        elif ovsentiment < 0:
            product.loc[i, 'sentipolarity'] = -1
        elif ovsentiment == 0:
            product.loc[i, 'sentipolarity'] = 0
        else:
            print('Something is wrong!!!')

    print('Update value of possentiment, negsentiment, and sentipolarity columns in prodpage table')
    for j in tqdm(range(len(product))):
        mongoid = '"' + product.loc[j, 'id'] + '"'
        possenti = str(product.loc[j, 'possentiment'])
        negsenti = str(product.loc[j, 'negsentiment'])
        sentipol = str(product.loc[j, 'sentipolarity'])
        sqlc1 = 'UPDATE prodpage SET '
        sqlc2 = 'possentiment = '
        sqlc3 = 'negsentiment = '
        sqlc4 = 'sentipolarity = '
        sqlc5 = 'WHERE id = '
        sqlall = sqlc1 + sqlc2 + possenti + ', ' + sqlc3 + negsenti + ', ' + \
                sqlc4 + sentipol + ' ' + sqlc5 + mongoid
        # update DB columns' value
        cursor.execute(sqlall)
        connection.commit()


##### one
#csvinput = './csvfiles/output1_reviewtext.csv'
#csvoutput = './csvfiles/output_englishstem.csv'
#englishStem(csvinput, conn, c, csvoutput)

##### two
csvinput = './csvfiles/output_englishstem.csv'
csvoutput = './csvfiles/output_sentiment.csv'
classSentiment(csvinput, csvoutput)

##### three
#addCol(conn, c)

##### four
#csvinput = './csvfiles/output_sentiment.csv'
#upReviewsTable(csvinput, conn, c)

##### five
#addColProdpage(conn, c)

##### six
#csvinput = './csvfiles/output_sentiment.csv'
#upProdpageTable(csvinput, conn, c)


conn.close()
