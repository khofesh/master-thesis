'''
cluster product sold based on:
    1. review count
    2. prodseen
        set '5a93e8768cbad97881597597' prodseen to 37000
'''
import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# make connection to sqlite db
conn = sqlite3.connect('product.db')
c = conn.cursor()

# enable foreign keys
c.execute("PRAGMA foreign_keys = ON")
conn.commit()

sqlc = 'SELECT id, ranking, reviewcount FROM prodpage'
c.execute(sqlc)
conn.commit()
prodpage = c.fetchall()
prodpage = pd.DataFrame(prodpage)
prodpage.columns = ['id', 'ranking', 'reviewcount']
# without 5aa39533ae1f941be7165ecd and 5aa2ad7735d6d34b0032a795
prodpage2 = prodpage.loc[2:,].copy()

# get the values
X = prodpage2.iloc[:, [1,2]].values
#X = prodpage.iloc[:, [1,2]].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# clustering
from sklearn.cluster import KMeans
wcss = []
for i in tqdm(range(1, 10)):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

## Elbow method
plt.plot(range(1, 10), wcss, 'bx-', marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# save the elbow plot
plt.show()

# applying k-means to the dataset
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# from 0-10
y_kmeans = kmeans.fit_predict(X)

# visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c='lime', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c='aquamarine', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c='navy', label='Cluster 3')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], c='black', label='Centroids')
plt.title('Clusters of ranking')
plt.xlabel('Ranking')
plt.ylabel('Review Count')
plt.legend()
plt.show()

prodpage2.index = range(0, len(prodpage2))
# add cluster to dataframe
prodpage2['cluster'] = pd.Series(y_kmeans)
# by default it starts from 0
prodpage2['cluster'] = prodpage2['cluster'] + 1
# save it to csv file
prodpage2['cluster'] = prodpage2['cluster'].astype('category')
prodpage2.to_csv('./csvfiles/salesclusters.csv', index=False)
