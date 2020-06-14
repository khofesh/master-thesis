# -*- coding: utf-8 -*-
import scrapy
from time import sleep
from selenium import webdriver
from scrapy.selector import Selector
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
import csv

class MinimalSpider(scrapy.Spider):
    name = 'minimal'
    allowed_domains = ['tokopedia.com']

    def start_requests(self):
        # open csv file
        with open('./csvfiles/produri.csv', newline='') as csvfile:
            record = csv.reader(csvfile, delimiter=',')
            for i in record:
                # [id, uri, helpfulreviewerscnt]
                yield scrapy.Request(i[0], callback=self.parse)

    def parse(self, response):
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
        options = webdriver.ChromeOptions()
        options.add_argument(f'user-agent={user_agent}')
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(chrome_options=options, 
                executable_path='/usr/bin/chromedriver')
        self.driver.get(response.request.url)

        # sleep for 10 seconds
        sleep(10)
        self.logger.info('Sleeping for 10 seconds')

        # extract source page
        selProdPage = Selector(text=self.driver.page_source)

        # Minimal beli
        minpurchase = selProdPage.xpath(
                '//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "pull-left m-0 border-none")]/text()'
                ).extract()[1]
        minpurchase = minpurchase.strip()

        yield {
                'uri':response.request.url,
                'minpurchase':minpurchase
                }
        
        self.driver.quit()
