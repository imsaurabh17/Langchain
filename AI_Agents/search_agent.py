import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

@tool
def get_weather(city: str) -> float:

    """This tool will provide the temperatur of a provided city"""

    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={WEATHER_API_KEY}"

    response = requests.get(url=geo_url)

    if response.status_code == 200:
        geo_data = response.json()
    
    geo_data = geo_data[0]
    lat = geo_data['lat']
    lon = geo_data['lon']

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"

    weather_response = requests.get(weather_url)

    if weather_response.status_code == 200:
        weather_data = weather_response.json()

    temperature = weather_data['main']['temp']

    return temperature

search_tool = DuckDuckGoSearchRun()

llm = ChatGroq(model="openai/gpt-oss-120b")

agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather]
)


if __name__=="__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Can you tell me about cristiano ronaldo and in which league he is currently playing and how many goals he has scored this season for each teams?"}]
    })

    print(result["messages"][-1].content)

    # print(get_weather.invoke({"city":"mumbai"}))