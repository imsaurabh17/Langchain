from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

prompt = PromptTemplate(
    template='Provide the score of LUL vs BIK match from the given {text}',
    input_variables=['text']
)

parser = StrOutputParser()

url = 'https://www.cricbuzz.com/'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))