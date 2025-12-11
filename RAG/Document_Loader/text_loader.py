from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('D:/Langchain Models/chats.txt',encoding='utf-8')

docs = loader.load()

model = ChatGroq(model='openai/gpt-oss-120b')

prompt = PromptTemplate(
    template='explain about the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'topic':docs[0].page_content}))