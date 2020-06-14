'''
how to run it:
    python ifhangs.py > 3_new.csv
'''
import pandas as pd

df1 = pd.read_csv('./output_minbeli2.csv')
df2 = pd.read_csv('./3.csv', header=None)
df2.columns = ['uri']

for i in range(len(df2)):
    if df2['uri'][i] not in list(df1['uri']):
        print(df2['uri'][i])
