import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults()


if __name__=="__main__":
    print(search_tool.invoke("current temperature in mumbai"))