from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

template1 = PromptTemplate(template="Write a detailed explaination about a {topic}",
                           input_variables=["topic"])

template2 = PromptTemplate(template="write a 3 line summary about the {text}")

parser = StrOutputParser()

chain = template1 | model | parser | model | parser

result = chain.invoke({"topic":"bermuda traingle"})

print(result)