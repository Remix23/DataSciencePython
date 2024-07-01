import requests as rq
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

URL = "https://www.pararius.com/english"

headers = [

]

browser = webdriver.Safari()

### fun with pararius
def get_for_city (url, city : str):

    browser.implicitly_wait

    url = f"{url}/apartments/{city.lower()}"

    browser.get(url=url)

    listings = browser.find_elements(By.CLASS_NAME, "search-list__item--listing")

    print(listings)

    time.sleep(20)

    browser.quit()

get_for_city(URL, "amsterdam")