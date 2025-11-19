from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=gemini_api_key)

vector = embeddings.embed_query("Who is ronaldo?")

print(vector)