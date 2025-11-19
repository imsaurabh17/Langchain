from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="openai/gpt-oss-120b",api_key=api_key)

result = model.invoke("What's the capital of India?")

print(result.content)