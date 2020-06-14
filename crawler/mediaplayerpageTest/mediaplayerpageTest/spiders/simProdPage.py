# -*- coding: utf-8 -*-
import scrapy
from time import sleep
from selenium import webdriver
from scrapy.selector import Selector
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from pymongo import MongoClient
import pymongo

# CRAWLER ini mengambil URLs pada database mongodb "urlproduct"
# execute clien-side javascript, iterate over reviews, 
# scrape items, and store it inside database mongodb "productpage"
class ProductPageSpider(scrapy.Spider):
    name = 'simpleproductpage'
    allowed_domains = ['tokopedia.com']
    start_urls = [
            'https://www.tokopedia.com/vaperslounge/druga-rda-black-100-authentic-by-augvape?trkid=f=Ca2090L000P0W0S0Sh00Co0Po0Fr0Cb0_src=directory_page=1_ob=5_q=_po=38_catid=2091'
            ]

    def __init__(self):
        # Connect to mongodb
        self.client = MongoClient('127.0.0.1', 27017)

    def parse(self, response):
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.119 Safari/537.36'
        options = webdriver.ChromeOptions()
        options.add_argument(f'user-agent={user_agent}')
        self.driver = webdriver.Chrome(chrome_options=options,
                executable_path='/usr/bin/chromedriver')
        self.driver.get(response.request.url)
        # sleep for 40 seconds
        sleep(40)
        self.logger.info('Sleeping for 40 seconds')

        urlCount = 1
        # get the "url", then load
        # the page
        selProdPage = Selector(text=self.driver.page_source)
            
        # extract needed items
        # rubah string URI menjadi list
        uriList = list(response.request.url)
        uriList = uriList[-6:-1] + list(uriList[-1])
        uriList = ''.join(uriList)
        # test whether the merchant used topads
        if uriList == 'topads':
            topads = 'yes'
        else:
            topads = 'no'
        # Kategori produk
        prodCat = selProdPage.xpath(
                '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
                ).extract()[0]
        # SubKategori produk
        prodSubCat = selProdPage.xpath(
                '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
                ).extract()[1]
        # SubSubkategori produk
        prodSubSubCat = selProdPage.xpath(
                '//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()'
                ).extract()[2]
        # Nama produk
        prodName = selProdPage.xpath(
                '//div[@id="breadcrumb-container"]/descendant::li/h2/text()'
                ).extract()[0]
        # Harga produk
        prodPrice = selProdPage.xpath(
                '//div[@class="product-box-content"]/descendant::span[@itemprop="price"]/text()'
                ).extract()[0].replace('.', '')
        # Nama merchant
        merchantName = selProdPage.xpath(
                '//div[@class="product-box-content"]/descendant::a[@id="shop-name-info"]/text()'
                ).extract()[0]
        # Gold merchant / biasa
        testMerchantType = selProdPage.xpath(
                '//div[@class="product-box-content"]/descendant::div/i/@data-original-title'
                ).extract()
        # Tes apakah biasa, gold, atau official
        if testMerchantType == '':
            merchantType = 'biasa'
        elif testMerchantType == 'Gold Merchant':
            merchantType = 'gold'
        else:
            merchantType = 'official'

        # Reputasi merchant
        merchantRep = selProdPage.xpath(
                '//div[@class="product-box-content"]/descendant::img/@data-original-title'
                ).extract()[0].replace(' points', '').replace('.', '')
        # Jumlah pengunjung
        prodSeen = selProdPage.xpath(
                '//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "view-count")]/text()'
                ).extract()[0]
        # Jumlah produk terjual
        prodSold = selProdPage.xpath(
                '//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "item-sold-count")]/text()'
                ).extract()[0]
        # Gambar produk
        prodPic = 'yes'
        # Rating produk / Nilai produk (1 - 5)
        prodRating = selProdPage.xpath(
                '//div[contains(@class, "reviewsummary-loop")]/div/p/text()'
                ).extract()[0]
        # Jumlah ulasan
        reviewCount = selProdPage.xpath(
                '//div[contains(@class, "reviewsummary-loop")]/descendant::div[@class="mt-5"]/text()'
                ).extract()[0].replace('\xa0', ' ').split(' ')[0]
        # Jumlah diskusi
        discussionCount = selProdPage.xpath(
                '//li[@id="p-nav-talk"]/a/span/text()'
                ).extract()[0]
        # Jumlah bintang
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
        try:
            # cashback
            cashback = selProdPage.xpath(
                    '//div[@class="product-box product-price-box mt-120"]/descendant::div[@class="product-content__cashback text-center fs-10"]/p/text()'
                    ).extract()[0]
            cashback = cashback.replace(' ', '').replace('\nDapatkancashback', '').replace('keTokoCash\n', '')
        except IndexError:
            cashback = 'no cashback'


        # sleep for 20 seconds
        sleep(10)
        self.logger.info('Sleeping for 10 seconds')

        ## connect to mongod database
        # "producturl" database
        self.db2 = self.client.producturl
        # collection named "prodpage"
        self.collection2 = self.db2.prodpage

        # insert scraped data into "prodpage"
        result = self.collection2.insert_one(
                {
                    'uri': response.request.url,
                    'topads': topads,
                    #'IDurlproductDB': self.response['_id'],
                    'prodName': prodName,
                    'prodPrice': prodPrice,
                    'prodCat': prodCat,
                    'prodSubCat': prodSubCat,
                    'prodSubSubCat': prodSubSubCat,
                    'merchantName': merchantName,
                    'merchantType': merchantType,
                    'merchantRep': merchantRep,
                    'prodSeen': prodSeen,
                    'prodSold': prodSold,
                    'prodPic': prodPic,
                    'prodRating': prodRating,
                    'reviewCount': reviewCount,
                    'discussionCount': discussionCount,
                    'cnt5star': cnt5star,
                    'cnt4star': cnt4star,
                    'cnt3star': cnt3star,
                    'cnt2star': cnt2star,
                    'cnt1star': cnt1star,
                    'cashback': cashback,
                    'helpfulReviewers': [
                        ],
                    'ordinaryReviewers': [
                        ],
                    'prodDiscussions': [
                        ]
                    }
                )

        # sleep for 20 seconds
        sleep(5)
        self.logger.info('sleeping for 5 seconds')

        # Klik tab Ulasan
        reviewTab = self.driver.find_element_by_xpath(
                '//li[@id="p-nav-review"]'
                )
        reviewClick = ActionChains(self.driver)
        reviewClick.move_to_element(reviewTab).click().perform()

        selProdPage2 = Selector(text=self.driver.page_source)

        # sleep for 40 seconds
        sleep(40)
        self.logger.info('Sleeping for 40 seconds')

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
        # Isi review
        commentGvnByReviewers = selProdPage2.xpath(
                '//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "most__helpful")]/div[@class="relative"]/p/text()'
                ).extract()

        # Smiley positif, negatif dan netral
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

        ## Yield ke mongodb
        # PERLU IF statement
        # untuk produk yang tidak
        # memiliki "ulasan paling membantu"
        if not imgReviewers:
            print('Nothing')
            self.collection2.update(
                    {'_id': result.inserted_id},
                    {'$addToSet': {'helpfulReviewers':
                        {'review': 'no review'}
                        }
                        }
                    )
            pass
        else:
            for index in range(len(imgReviewers)):
                self.collection2.update(
                        {'_id': result.inserted_id},
                        {'$addToSet': {'helpfulReviewers':
                            {'imgReviewers': imgReviewers[index],
                                'ratingGvnByReviewers': ratingGvnByReviewers[index],
                                'commentGvnByReviewers': commentGvnByReviewers[index],
                                'posSmileyReviewers': posSmileyReviewers[index],
                                'neutSmileyReviewers': neutSmileyReviewers[index],
                                'negSmileyReviewers': negSmileyReviewers[index]
                                }
                            }
                            }
                        )

        # sleep for 10 seconds
        sleep(5)
        self.logger.info('sleeping for 5 seconds')

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

        # insert ordinary reviewers into mongodb database
        # into "ordinaryReviewers" array
        if not imgReviewersOrd:
            print('Nothing')
            self.collection2.update(
                    {'_id': result.inserted_id},
                    {'$addToSet': {'helpfulReviewers':
                        {'review': 'no review'}
                        }
                        }
                    )
            pass
        else:
            for index in range(len(imgReviewersOrd)):
                self.collection2.update(
                        {'_id': result.inserted_id},
                        {'$addToSet': {'ordinaryReviewers':
                            {'imgReviewersOrd': imgReviewersOrd[index],
                                'ratingGvnByReviewersOrd': ratingGvnByReviewersOrd[index],
                                'commentGvnByReviewersOrd': commentGvnByReviewersOrd[index],
                                'reviewOrdHelpful': reviewOrdHelpful[index],
                                'posSmileyReviewersOrd': posSmileyReviewersOrd[index],
                                'neutSmileyReviewersOrd': neutSmileyReviewersOrd[index],
                                'negSmileyReviewersOrd': negSmileyReviewersOrd[index]
                                }
                            }
                            }
                        )

        # sleep for 10 seconds
        sleep(5)
        self.logger.info('sleeping for 5 seconds')

        # find ">", then click it
        while True:
            try:
                # find ">" symbol, then click it
                nextDiscussion = self.driver.find_element_by_xpath(
                        '//div[@class="pagination pull-right"]/descendant::i[@class="icon-chevron-right"]'
                        )
                actions = ActionChains(self.driver)
                actions.move_to_element(nextDiscussion).click().perform()
                
                # save page source
                selReviews = Selector(text=self.driver.page_source)
                # sleep for 10 seconds
                sleep(10)
                self.logger.info('sleeping for 10 seconds')

                # scrape ordinary reviewers
                # Gambar Reviewer biasa/ordinary
                imgReviewersOrd = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::img[@class="list-box-image"]/@src'
                        ).extract()
                # Rating yg diberikan oleh 
                # Reviewer biasa
                ratingGvnByReviewersOrd = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-text")]/div/i/@class'
                        ).extract()
                # Dibersihkan:
                for i in range(len(ratingGvnByReviewersOrd)):
                    ratingGvnByReviewersOrd[i] = ratingGvnByReviewersOrd[i].replace(
                            'rating-star rating-star', ''
                            )
                # Isi review
                commentGvnByReviewersOrd = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-text")]/span[@class="review-body"]/text()'
                        ).extract()
                # Number of people who find ordinary reviewers comment helpful
                reviewOrdHelpful = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::div[@class="like-review"]/div/text()'
                        ).extract()
                # Dibersihkan:
                for i in range(len(reviewOrdHelpful)):
                    reviewOrdHelpful[i] = reviewOrdHelpful[i].replace(
                            ' orang lainnya terbantu dengan ulasan ini', ''
                            )
                # Smiley reviewer biasa
                # Positive
                posSmileyReviewersOrd = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "green")]/text()'
                        ).extract()
                # Neutral
                neutSmileyReviewersOrd = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "yellorange")]/text()'
                        ).extract()
                # Negative
                negSmileyReviewersOrd = selReviews.xpath(
                        '//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "red")]/text()'
                        ).extract()

                # insert ordinary reviewers into mongodb database
                # into "ordinaryReviewers" array
                if not imgReviewersOrd:
                    print('Nothing')
                    self.collection2.update(
                            {'_id': result.inserted_id},
                            {'$addToSet': {'ordinaryReviewers':
                                {'review': 'no review'}
                        }
                        }
                    )
                    pass
                else:
                    for index in range(len(imgReviewers)):
                        self.collection2.update(
                                {'_id': result.inserted_id},
                                {'$addToSet': {'ordinaryReviewers':
                                    {'imgReviewersOrd': imgReviewersOrd[index],
                                        'ratingGvnByReviewersOrd': ratingGvnByReviewersOrd[index],
                                        'commentGvnByReviewersOrd': commentGvnByReviewersOrd[index],
                                        'reviewOrdHelpful': reviewOrdHelpful[index],
                                        'posSmileyReviewersOrd': posSmileyReviewersOrd[index],
                                        'neutSmileyReviewersOrd': neutSmileyReviewersOrd[index],
                                        'negSmileyReviewersOrd': negSmileyReviewersOrd[index]
                                        }
                                    }
                                    }
                                )

                # sleep for 10 seconds
                sleep(10)
                self.logger.info('sleeping for 10 seconds')

            except NoSuchElementException:
                self.logger.info('no more discussion to load')
                break

        # click "Diskusi Produk"
        discussTab = self.driver.find_element_by_xpath(
                '//li[@id="p-nav-talk"]'
                )
        discussClick = ActionChains(self.driver)
        discussClick.move_to_element(discussTab).click().perform()

        # extract page source
        selDiscuss = Selector(text=self.driver.page_source)

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

        # insert number of questions and answers into mongodb database
        # into "diskusiProduk" array
        if questionsCnt == 0:
            print('No Discussion')
            self.collection2.update(
                    {'_id': result.inserted_id},
                    {'$addToSet': {'prodDiscussions':
                        {'discussion': 'no discussion'}
                        }
                        }
                    )
            pass
        else:
            while True:
                try:
                    # click ">"
                    nextDiscussion2 = self.driver.find_element_by_xpath(
                            '//div[@class="pagination pull-right"]/descendant::i[@class="icon-chevron-right"]'
                            )
                    actionDisc = ActionChains(self.driver)
                    actionDisc.move_to_element(nextDiscussion2).click().perform()

    
                    # extract page source
                    selDiscuss2 = Selector(text=self.driver.page_source)
    
                    # add number of questions
                    questionsCnt += len(selDiscuss2.xpath(
                        '//li[@total-items="talksTotal"]/input[@name="total_comment"]/@value'
                        ).extract())
                    # add number of answers
                    answersCnt += len(selDiscuss2.xpath(
                        '//li[@total-items="talksTotal"]/input[@name="total_comment"]/@value'
                        ).extract())
                    # extract jumlah comment masing-masing pertanyaan
                    testDiscuss = selDiscuss2.xpath(
                            '//li[@total-items="talksTotal"]/input[@name="total_comment"]/@value'
                            ).extract()
                    # jika ada yang == '0', maka kurangi jumlah
                    # jawaban
                    for i in testDiscuss:
                        if i == '0':
                            answersCnt -= 1
                        else:
                            pass

                    # sleep for 10 seconds
                    sleep(5)
                    self.logger.info('sleeping for 5 seconds')

                except NoSuchElementException:
                    self.collection2.update(
                            {'_id': result.inserted_id},
                            {'$addToSet': {'prodDiscussions':
                                {'questionsCnt': questionsCnt,
                                    'answersCnt': answersCnt
                                    }
                                }
                                }
                            )
                    self.driver.quit()
                    self.client.close()
                    break
