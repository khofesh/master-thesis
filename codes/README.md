## Steps followed to build the model

## Folders :
    1. crawler: scrapy source code
    2. csvfiles: 
    3. defaultavatar:
    4. lib:
    5. graphs:
    6. profpics: 

0. Crawler

1. Convert all reviews from mongodb database. Take only product that has
   reviews >= 5 (for ordinary reviewers).
```
   $ python mongotosqlite.py
```
2. Download profile pictures.
```
   $ python reviewsdownpics.py
```
3. Compare profile pictures and default avatar.
 ``` 
   $ python reviewscompics.py
```
4. Fix ValueError on certain products manually.
```
   $ python pfixerror.py
```
5. Change date format in database from "03 Mar 2018" -> "2018-03-03" (Year-mm-d)
```
   $ python pdate.py
```
6. Run crawler to check whether helpful review contains product pictures in it.
```
   $ cd ./hasprodpic/hasprodpic/spider/
   $ scrapy hasprodpic -o output.csv
```
7. Delete duplicate uri in prodpage
```
   $ python pduplicate.py
   $ python pdelduplicate.py
```
8. Update helpful reviewers prodpics value
```
   $ python phrprodpics.py
```
9. Delete row in database that's not in "Elektronik " category
```
   $ python delrecord.py
```
10. Cleaning and preparing text data
```
   $ python pcsvforjava.py
   $ python pcsvforjava2.py
   $ cp csvfiles/input_reviewtext.csv lib/textpreprocessing/
   $ cd lib/textpreprocessing/
   $ sbt
   > run (choose 1), then
   > run (choose 2)
```
11. Sentiment analysis and english language stemming
```
   $ cp lib/textpreprocessing/output1_reviewtext.csv csvfiles/

   # Stem english language
   $ vim psentiment.py
   	uncomment three lines below ##### one
   $ python psentiment.py

   # Sentiment
   $ vim psentiment.py
   	uncomment three lines below ##### two
   $ python psentiment.py

   # Add column to reviews table
   $ vim psentiment.py
   	uncomment one line below ##### three
   $ python psentiment.py

   # Update reviews table
   $ vim psentiment.py
   	uncomment two lines below ##### four
   $ python psentiment.py

   # Add column to prodpage table
   $ vim psentiment.py
   	uncomment one line below ##### five
   $ python psentiment.py
   
   # Update prodpage table (possentiment, negsentiment, sentipolarity)
   $ vim psentiment.py
   	uncomment two lines below ##### six
   $ python psentiment.py
```
12. Create subfolder inside profpics folder, then move .jpg file to its matching id.
    Subfolder is named after mongodb '_id'
```
   $ python pmkdirmv.py
```
13. Complete all the variables needed to build a model
```
   $ python pvariable.py
```
14. Create .gz archive of profpics
```
   $ tar -czvf profpics.tgz ./profpics/
```
15. Extract id, uri, merchantname, and merchanttype. Manually 
    correct merchanttype (copy paste uri into browser).
    faulty logic in scraper.
```
   $ python ptrivial.py
   # update sqlite db
   $ python ptrivial2.py
```
16. Clustering
```
   $ python sales_clustering.py
```
17. create salescluster column in database and update it
```
   $ python pvariable.py
```
18. generate output_training.csv for the upteenth time :)
```
   $ python pvariable.py
```
19. add 'training' table inside product.db
```
   uncomment:
   	trainingTable(conn, c)
   	upTrainTable(conn, c, 'training')
   $ python pvariable.py
```
20. compiling C source code
```
   $ cd 'grangertest/2006_GC_JEDC_c_and_exe_code/GCtest C and MS Command Prompt'
   add '#define _GNU_SOURCE' inside GCTtest.c
   $ gcc GCTtest.c -o GCTtest -lm -std=gnu99
```
21. edit prodsold of 5a904746f97c5d2c89d4c62f and 5aa772fb35d6d35ade366d47 from 1000 -> 10000

22. Model training (ranking, sentiment, and forecasting)

23. validasi
```
   $ cd crawler
   $ scrapy startproject validasi
   $ scrapy genspider tokopedia tokopedia.com
   $ cd ../../../../
   $ vim createUrlForValidation.py
   $ python createUrlForValidation.py
   $ cd crawler/validasi/validasi/spiders
   $ split urlValidasi.csv -l 1339
   $ cd ../../../../
   $ cp -R validasi/ /media/ext4/container/three/home/fahmi3/
   $ cp -R validasi/ /media/ext4/container/two/home/fahmi2/
```
24. convert mongodb 'validasi' into sqlite3
```
   $ python validasisqlite.py
   -> If there is an error that said:
   sqlite3.IntegrityError: UNIQUE constraint failed
   -> delete all rows in database, then start over
```
25. create csv file for sentiment analysis validation
```
   $ python validasiCsvSentiment.py

   $ cp csvfiles/input_reviewtext_validasi.csv lib/textpreprocessing/input_reviewtext.csv
   $ cd lib/textpreprocessing/
   $ sbt
   > run (choose 1), then
   $ cd ../../

   $ cp lib/textpreprocessing/output1_reviewtext.csv csvfiles/output1_reviewtext_validasi.csv

   # Stem english language
   $ vim validasiSentiment.py
   	uncomment lines below '##### one', '##### two', three, four, five, and six
   $ python validasiSentiment.py
```
26. validasi model sentiment analysis
```
   validasiModelSentimen.py
```
27. correct id in validasi.db with id in product.db by matching their uri
```
   validasiCorrectID.ipynb
```
28. add required columns into table 'prodpage' and 'reviews' of validasi.db
```
   validasiAddColumns.ipynb
```
29. Training salescluster classification
```
   groupClassification.ipynb
```
30. Add more salescluster == 2.0 into the database
```
   validasiAddMoreGroup2.ipynb
   $ cp ./crawler/validasi/validasi/spiders/urlValidasi2.csv 
   	/media/ext4/container/two/home/fahmi2/validasi/validasi/spiders/urlValidasi.csv
   ```
   do web scraping, again!

31. validasi model forecasting
```
   validasiModelForecasting.ipynb
   $ python validasiModelForecasting.py
```
32. validasi model group class/salescluster
```
   $ python validasiModelGroupClass.py
 ```
33. validasi model ranking
   ```
   $ python validasiModelRanking.py
```
34. perbaiki model sentimen dan ranking

35. otomasi pembuatan model peramalan
   - model neural network multivariat untuk data yang imputed or not imputed
    tseriesNN.py
   - model multiple linear regression untuk data yang imputed or not imputed
    tseriesLR.py
   - model SVM multivariat untuk data yang imputed or not imputed
    tseriesSVR.py
   - model neural network univariat untuk data yang imputed or not imputed
    tseriesNN_univariate.py
   - model simple linear regression untuk data yang imputed or not imputed
    tseriesLR_univariate.py
   - model SVM univariat untuk data yang imputed or not imputed
    tseriesSVR_univariate.py

36. 

