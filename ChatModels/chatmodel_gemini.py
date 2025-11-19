from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro", api_key=gemini_api_key)

result = model.invoke("Who is Cristiano Ronaldo?")

print(result.content)