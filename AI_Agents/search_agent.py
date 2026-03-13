import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

search_tool = DuckDuckGoSearchRun()

llm = ChatGroq(model="openai/gpt-oss-120b")

agent = create_react_agent(
    model=llm,
    tools=[search_tool]
)


if __name__=="__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is the current temperature in Mumbai?"}]
    })

    print(result["messages"][-1].content)