from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is Cristiano Ronaldo?")

print(result.content)