from bs4 import BeautifulSoup as bs
import pandas as pd
import time, requests

url = "https://www.sendo.vn/tu-ke-tui-dung-giay"

html_content = requests.get(url).text
soup = bs(html_content, "html.parser")
print(soup.title.text)

for link in soup.find_all(class_="Root_1Kcx"):
    div = link.find("div", {"id": "page-container"})
    div_result = div.find("div", {"class": "resultPanel_25EW"})
    for item in div_result.find_all("div"):
        print(item)
