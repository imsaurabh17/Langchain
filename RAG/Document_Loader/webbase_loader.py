from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

url = 'https://www.cricbuzz.com/'

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)