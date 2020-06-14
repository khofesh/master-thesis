# -*- coding: utf-8 -*-
import scrapy
from time import sleep
from selenium import webdriver
from scrapy.selector import Selector
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains

## EXPERIMENT WITH CHROME HEADLESS ##
class MediaplayerSpider(scrapy.Spider):
    name = 'scrapeuri'
    allowed_domains = ['tokopedia.com']
    start_urls = ['https://www.tokopedia.com/p/elektronik/perangkat-elektronik-lainnya?ob=5']
#sudah            ['https://www.tokopedia.com/p/elektronik/vaporizer?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/audio?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/tv?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/kamera-pengintai?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/media-player?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/telepon?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/tool-kit?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/pencahayaan?ob=5',
#sudah            'https://www.tokopedia.com/p/elektronik/listrik?ob=5',
#            'https://www.tokopedia.com/p/elektronik/perangkat-elektronik-lainnya?ob=5']

    def parse(self, response):
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
        options = webdriver.ChromeOptions()
        # specify headless mode
        #options.add_argument('headless')
        # specify the desired user agent
        options.add_argument(f'user-agent={user_agent}')

        self.driver = webdriver.Chrome(chrome_options=options,
                executable_path='/usr/bin/chromedriver')
        self.driver.maximize_window()
        self.driver.get(response.request.url)
        sleep(30)
        try:
            self.driver.find_element_by_xpath(
                    '//*[@class="hopscotch-bubble-container"]/a'
                    ).click()
        except NoSuchElementException:
            pass

        sleep(20)
        self.driver.find_element_by_xpath(
                '//div[@id="hopscotch-filter"]'
                ).click()
        try:
            self.driver.find_element_by_xpath(
                    '//div[@id="hopscotch-filter"]/descendant::div[@class="sort-intermediary"]/select[@id="order-by"]/option[@value="5"]'
                    ).click()
        except NoSuchElementException:
            pass

        sleep(20)

        pagenumber = 1
        while True:
            try:
                sel = Selector(text=self.driver.page_source)
                sleep(4)
                self.logger.info('Sleeping for 3 seconds')

                urlProdTopads = sel.xpath(
                        '//*[@class="ta-slot-card"]/a[not(@class="hide")]/@href'
                        ).extract()
                urlProdBiasa = sel.xpath(
                        '//*[contains(@class, "category-product-box")]/a/@href'
                        ).extract()

                urlNumber = 1

                for i in urlProdTopads:
                    yield {"Topads": "yes",
                            "url": i,
                            "urlNumber": urlNumber
                            }
                    urlNumber += 1

                for j in urlProdBiasa:
                    yield {"Topads": "no",
                            "url": j,
                            "urlNumber": urlNumber
                            }
                    urlNumber += 1

                # check whether the page is multiple 10 or not
                # when it is, rest for 80 seconds
                # then continue
                if pagenumber % 2 == 0 and pagenumber % 10 == 0:
                    print(f'Halaman {pagenumber}, berhenti setiap halaman kelipatan 10')
                    sleep(80)
                    self.logger.info('Sleeping for 60 seconds')
                                
                # stop when page number = 50
                #if pagenumber == 50:
                #    break
            
                next_page = self.driver.find_element_by_xpath(
                        '//li[@class="ng-scope"]/a[@ng-click="filter.page = pagination.current_page + 1; filter_changed(filter.page,false)"]'
                        )
                actions = ActionChains(self.driver)
                actions.move_to_element(next_page).click().perform()
                #next_page.click()
                sleep(20)
                self.logger.info('Sleeping for 20 seconds')

                pagenumber += 1

            except NoSuchElementException:
                self.logger.info('No more pages to load')
                self.driver.quit()
                break

