import requests as rq
from bs4 import BeautifulSoup
from selenium import webdriver

URL = "https://www.pararius.com/english"

headers = [

]

### fun with pararius
def get_for_city (url, city : str):

    url = f"{url}/apartments/{city.lower()}" 

    raw = rq.get(url = url)

    print(raw.text)

    if raw.status_code != 200: return -1

    soup = BeautifulSoup(raw.content, "html.parser")

    res = soup.find_all("section", class_ = "listing-search-item")

    print(res)


