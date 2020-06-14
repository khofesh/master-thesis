import scrapy
from time import sleep
from selenium import webdriver
from scrapy.selector import Selector
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from pymongo import MongoClient
import pymongo
import csv

user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.119 Safari/537.36'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
options = webdriver.ChromeOptions()
options.add_argument(f'user-agent={user_agent}')
driver = webdriver.Chrome(chrome_options=options,executable_path='/usr/bin/chromedriver')
driver.get(response.request.url)

selProdPage = Selector(text=driver.page_source)

###############################################

uriList = list(response.request.url)
uriList = uriList[-6:-1] + list(uriList[-1])
uriList = ''.join(uriList)

## test whether the merchant used topads
if uriList == 'topads':
    topads = 'yes'
else:
    topads = 'no'

## Kategori produk
# IndexError, perbaiki
prodCat = selProdPage.xpath(
        '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
        ).extract()[0]
## SubKategori produk
prodSubCat = selProdPage.xpath(
        '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
        ).extract()[1]
## SubSubkategori produk
# IndexError
try:
    prodSubSubCat = selProdPage.xpath(
            '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
            ).extract()[2]
except IndexError:
    print('prodSubSubCat IndexError')
    print(response.request.url)
    lenSubSubCat = len(selProdPage.xpath(
        '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
        ))
    prodSubSubCat = selProdPage.xpath(
            '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
            ).extract()[lenSubSubCat-1]

## Nama produk
prodName = selProdPage.xpath(
        '//div[@id="breadcrumb-container"]/descendant::li/h2/text()'
        ).extract()[0]
## Harga produk
prodPrice = selProdPage.xpath(
        '//div[@class="product-box-content"]/descendant::span[@itemprop="price"]/text()'
        ).extract()[0].replace('.', '')
## Nama merchant
merchantName = selProdPage.xpath(
        '//div[@class="product-box-content"]/descendant::a[@id="shop-name-info"]/text()'
        ).extract()[0]
## Gold merchant / biasa
merchantType = selProdPage.xpath(
        '//div[@class="product-box-content"]/descendant::div/i/@data-original-title'
        ).extract()

if merchantType:
    print('merchantType: {0}'.format(merchantType))
else:
    merchantType = 'biasa'
    print('merchantType: {0}'.format(merchantType))

## Reputasi merchant
merchantRep = selProdPage.xpath(
        '//div[@class="product-box-content"]/descendant::img/@data-original-title'
        ).extract()[0].replace(' points', '').replace('.', '')
## Jumlah pengunjung
prodSeen = selProdPage.xpath(
        '//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "view-count")]/text()'
        ).extract()[0]
## Jumlah produk terjual
prodSold = selProdPage.xpath(
        '//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "item-sold-count")]/text()'
        ).extract()[0]
## Gambar produk
prodPic = 'yes'
# Rating produk / Nilai produk (1 - 5)
# beberapa URL gagal diproses, IndexError
# karena beberapa produk belum dirating oleh konsumen
try:
    prodRating = selProdPage.xpath('//div[contains(@class, "reviewsummary-loop")]/div/p/text()').extract()[0]
except IndexError:
    print('prodRating IndexError:')
    print(response.request.url)
    # dari prodRating sebelumnya, hasilnya bisa berupa '[]'
    prodRating = 'no rating'

# there were cases when it throws IndexError on both of reviewCount 
# and discussionCount
# so, try-except is used to handle this error
try:
    # Jumlah ulasan
    # IndexError
    reviewCount = selProdPage.xpath(
            '//li[@id="p-nav-review"]/a/span/text()'
            ).extract()[0]
except IndexError:
    print('reviewCount IndexError:')
    print(response.request.url)
    reviewCount = '0'

try:
    # Jumlah diskusi
    # IndexError
    discussionCount = selProdPage.xpath(
            '//li[@id="p-nav-talk"]/a/span/text()'
            ).extract()[0]
except IndexError:
    print('discussionCount IndexError:')
    print(response.request.url)
    discussionCount = '0'

## Jumlah bintang
# IndexError
try:
    # Jumlah bintang 5
    cnt5star = selProdPage.xpath(
            '//div[contains(@class, "ratingtotal")]/text()'
            ).extract()[0].replace(' ','').replace('\n', '')
    # Jumlah bintang 4
    cnt4star = selProdPage.xpath(
            '//div[contains(@class, "ratingtotal")]/text()'
            ).extract()[1].replace(' ','').replace('\n', '')
    # Jumlah bintang 3
    cnt3star = selProdPage.xpath(
            '//div[contains(@class, "ratingtotal")]/text()'
            ).extract()[2].replace(' ','').replace('\n', '')
    # Jumlah bintang 2
    cnt2star = selProdPage.xpath(
            '//div[contains(@class, "ratingtotal")]/text()'
            ).extract()[3].replace(' ','').replace('\n', '')
    # Jumlah bintang 1
    cnt1star = selProdPage.xpath(
            '//div[contains(@class, "ratingtotal")]/text()'
            ).extract()[4].replace(' ','').replace('\n', '')
except IndexError:
    print('Jumlah bintang IndexError')
    print(response.request.url)
    cnt5star = '0'
    cnt4star = '0'
    cnt3star = '0'
    cnt2star = '0'
    cnt1star = '0'

## cashback
try:
    # cashback
    cashback = selProdPage.xpath(
            '//div[@class="product-box product-price-box mt-120"]/descendant::div[@class="product-content__cashback text-center fs-10"]/p/text()'
            ).extract()[0]
    cashback = cashback.replace(' ', '').replace('\nDapatkancashback', '').replace('keTokoCash\n', '')
except IndexError:
    print('no cashback')
    cashback = 'no cashback'

## Klik tab Ulasan
reviewTab = driver.find_element_by_xpath(
        '//li[@id="p-nav-review"]'
        )
reviewClick = ActionChains(driver)
reviewClick.move_to_element(reviewTab).click().perform()

selProdPage2 = Selector(text=driver.page_source)

## Ulasan Paling Membantu
# ada yang tidak terdapat "ulasan paling membantu"
# Gambar reviewers yang paling membantu
# [] empty list evaluate to False in boolean contexts
imgReviewers = selProdPage2.xpath(
        '//div[contains(@class, "multiple-mosthelpful")]/descendant::img[@class="list-box-image"]/@src'
        ).extract()
# Rating yg diberikan reviewers yg paling membantu
ratingGvnByReviewers = selProdPage2.xpath(
        '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "most__helpful")]/span/@class'
        ).extract()
# dibersihkan
for i in range(len(ratingGvnByReviewers)):
    ratingGvnByReviewers[i] = ratingGvnByReviewers[i].split(' '
            )[0].replace('rating-star', '')
## Review text
commentGvnByReviewers = selProdPage2.xpath(
        '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "most__helpful")]/div[@class="relative"]/p/text()'
        ).extract()
## Smiley positif, negatif dan netral
# Positive Smiley
posSmileyReviewers = selProdPage2.xpath(
        '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "green")]/text()'
        ).extract()
# Neutral Smiley
neutSmileyReviewers = selProdPage2.xpath(
        '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "yellorange")]/text()'
        ).extract()
# Negative Smiley
negSmileyReviewers = selProdPage2.xpath(
        '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "red")]/text()'
        ).extract()
# Added try-except, just in case it throws Error
try:

    ## Tanggal dan Jam
    dTR = selProdPage2.xpath(
            '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "most__helpful")]/descendant::small[contains(@class, "muted")]/i/text()'
            ).extract()
    # Tanggal review
    dateReviewers = []
    # Jam review
    timeReviewers = []
    # assign date to dateReviewers
    # and assign time to timeReviewers
    for i in dTR:
        tmp = i.split(',')
        dateReviewers.append(tmp[0])
        timeReviewers.append(tmp[1].strip())
except IndexError:
    dateReviewers = 'NaN'
    timeReviewers = 'NaN'

## Review biasa
# Gambar Reviewer biasa/ordinary
imgReviewersOrd = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::img[@class="list-box-image"]/@src'
        ).extract()
# Rating yg diberikan oleh 
# Reviewer biasa
ratingGvnByReviewersOrd = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-text")]/div/i/@class'
        ).extract()
# Dibersihkan:
for i in range(len(ratingGvnByReviewersOrd)):
    ratingGvnByReviewersOrd[i] = ratingGvnByReviewersOrd[i].replace(
            'rating-star rating-star', ''
            )
# Isi review
commentGvnByReviewersOrd = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-text")]/span[@class="review-body"]/text()'
        ).extract()
# Number of people who find ordinary reviewers comment helpful
reviewOrdHelpful = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::div[@class="like-review"]/div/text()'
        ).extract()
# Dibersihkan:
for i in range(len(reviewOrdHelpful)):
    reviewOrdHelpful[i] = reviewOrdHelpful[i].replace(
            ' orang lainnya terbantu dengan ulasan ini', ''
            )

# Smiley reviewer biasa
# Positive
posSmileyReviewersOrd = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "green")]/text()'
        ).extract()
# Neutral
neutSmileyReviewersOrd = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "yellorange")]/text()'
        ).extract()
# Negative
negSmileyReviewersOrd = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "red")]/text()'
        ).extract()
# Product pics attached to the review:
prodPicRevTest = selProdPage2.xpath(
        '//ul[@id="review-container"]/descendant::ul[contains(@class, "clearfix list-attachment")]'
        ).extract()
prodPicInReview = []
for i in prodPicRevTest:
    if "thumb mb-15" in i:
        prodPicInReview.append('yes')
    else:
        prodPicInReview.append('no')
# added try-except, just in case it throws Error
try:
    ## Tanggal dan Jam reviewer biasa
    dTROrd = selProdPage2.xpath(
            '//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-top")]/descendant::small[contains(@class, "muted")]/i/text()'
            ).extract()
    # Tanggal review reviewer biasa
    dateReviewersOrd = []
    # Jam review reviewer biasa
    timeReviewersOrd = []
    # assign date to dateReviewersOrd
    # assign time to timeReviewersOrd
    for i in dTROrd:
        tmp = i.split(',')
        dateReviewersOrd.append(tmp[0])
        timeReviewersOrd.append(tmp[1].strip())
except IndexError:
    dateReviewersOrd = 'NaN'
    timeReviewersOrd = 'NaN'

# click "Diskusi Produk"
discussTab = driver.find_element_by_xpath(
        '//li[@id="p-nav-talk"]'
        )
discussClick = ActionChains(driver)
discussClick.move_to_element(discussTab).click().perform()

# extract page source
selDiscuss = Selector(text=driver.page_source)

# number of questions
questionsCnt = len(selDiscuss.xpath(
    '//li[@total-items="talksTotal"]/input[@name="total_comment"]/@value'
    ).extract())
# number of answers
answersCnt = len(selDiscuss.xpath(
    '//li[@total-items="talksTotal"]/input[@name="total_comment"]/@value'
    ).extract())
# ekstrak jumlah comment masing2 pertanyaan
# output:
# ['3', '1', '1', '1', '3', '1', '1', '0', '7', '1']
testDiscuss = selDiscuss.xpath(
        '//li[@total-items="talksTotal"]/input[@name="total_comment"]/@value'
        ).extract()
# jika ada yang bernilai == '0', maka kurangi answerCnt 1
for i in testDiscuss:
    if i == '0':
        answersCnt -= 1
    else:
        pass

