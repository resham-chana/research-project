import requests

URL = "https://www.cia.gov/the-world-factbook/countries"
page = requests.get(URL)

print(page.text)