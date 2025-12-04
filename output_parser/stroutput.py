from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

template1 = PromptTemplate(template="Write a detailed discussion on {topic}",
                           input_variables=['topic'])

template2 = PromptTemplate(template="Write a 5 line summary on {text}",
                           input_variables=['text'])

prompt1 = template1.invoke({"topic":"Bermuda Triangle"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result1 = model.invoke(prompt2)

print(result1.content)