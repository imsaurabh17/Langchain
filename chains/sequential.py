from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

model = ChatGroq(model='openai/gpt-oss-20b')

prompt1 = PromptTemplate(
    template="Generate a detailed info about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Provide the most important 5 pointers from {output}",
    input_variables=['output']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'cr7'})

print(result)