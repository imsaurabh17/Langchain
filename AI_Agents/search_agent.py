import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

search_tool = DuckDuckGoSearchResults()

llm = ChatGroq(model="llama-3.3-70b-versatile")


if __name__=="__main__":
    print(search_tool.invoke("current temperature in mumbai"))