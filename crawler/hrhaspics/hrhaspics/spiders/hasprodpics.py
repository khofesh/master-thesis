# -*- coding: utf-8 -*-
import scrapy
from time import sleep
from selenium import webdriver
from scrapy.selector import Selector
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
import csv


class HasprodpicsSpider(scrapy.Spider):
    name = 'hasprodpics'
    allowed_domains = ['tokopedia.com']

    def start_requests(self):
        # open csv file
        with open('./productHasHelpfulReviewers.csv', newline='') as csvfile:
            record = csv.reader(csvfile, delimiter=',')
            for i in record:
                # [id, uri, helpfulreviewerscnt]
                yield scrapy.Request(i[1], callback=self.parse)

    def parse(self, response):
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
        options = webdriver.ChromeOptions()
        options.add_argument(f'user-agent={user_agent}')
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(chrome_options=options, 
                executable_path='/usr/bin/chromedriver')
        self.driver.get(response.request.url)

        # sleep for 20 seconds
        sleep(20)
        self.logger.info('Sleeping for 20 seconds')

        # extract source page
        selProdPage = Selector(text=self.driver.page_source)

        # helpful reviews content
        reviewContent = selProdPage.xpath(
                '//div[@id="helpful-review"]/descendant::div[contains(@class, "multiple-mosthelpful")]/div[@class="item clearfix mb-20"]'
                )

        picreview = []
        if len(reviewContent) == 3:
            review1 = reviewContent[0].extract()
            review2 = reviewContent[1].extract()
            review3 = reviewContent[2].extract()
            for i in [review1, review2, review3]:
                if "clearfix list-attachment mt-15" in i:
                    picreview.append(1)
                else:
                    picreview.append(0)
            # Yield
            yield {'uri':response.request.url,
                    'picreview1':picreview[0],
                    'picreview2':picreview[1],
                    'picreview3':picreview[2]
                    }
        elif len(reviewContent) == 2:
            review1 = reviewContent[0].extract()
            review2 = reviewContent[1].extract()
            for i in [review1, review2]:
                if "clearfix list-attachment mt-15" in i:
                    picreview.append(1)
                else:
                    picreview.append(0)
            # Yield
            yield {'uri':response.request.url,
                    'picreview1':picreview[0],
                    'picreview2':picreview[1],
                    'picreview3':0
                    }
        else:
            review1 = reviewContent.extract()
            if "clearfix list-attachment mt-15" in review1:
                picreview.append(1)
            else:
                picreview.append(0)
            # Yield
            yield {'uri':response.request.url,
                    'picreview1':picreview[0],
                    'picreview2':0,
                    'picreview3':0
                    }
        
        self.driver.quit()
