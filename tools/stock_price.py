import os
import requests
from typing import Annotated
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_groq import ChatGroq

load_dotenv()

API_KEY = os.getenv("EXCHANGE_RATE_API")

@tool
def get_stock_price(stock_symbol: str) -> float:
    """This tool will provide the live price of a stock"""
    ticker = yf.Ticker(stock_symbol)

    current_price = ticker.info.get("currentPrice")

    return current_price

@tool
def get_exchange_rate(source_currency: str, target_currency: str) -> float:
    """This tool will provide the exchange rate of a currency from source to target"""

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{source_currency}/{target_currency}"

    response = requests.get(url)

    if response.status_code==200:
        data = response.json()
        conversion_rate = data["conversion_rate"]
        return conversion_rate
    else:
        return "API Failed to fetch the conversion rate"
    
@tool
def calculate_investment(share: int, price: float, rate: float) -> float:
    """This tool will provide the total investment"""

    investment = share * price * rate

    return investment

llm = ChatGroq(model="llama-3.3-70b-versatile")

tools = [get_stock_price,get_exchange_rate,calculate_investment]

llm_with_tools = llm.bind_tools(tools)


if __name__=="__main__":
    messages = []
    stock_price = get_stock_price.invoke({"stock_symbol":"AAPL"})
    rate = get_exchange_rate.invoke({"source_currency":"USD","target_currency":"INR"})
    #print(calculate_investment.invoke({"share":10,"price":stock_price,"rate":rate}))
    query = HumanMessage("what is the cost of 10 shares of AAPL in INR?")
    messages.append(query)
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    for tool_call in ai_message.tool_calls:

        if tool_call["name"] == "get_stock_price":
            output = get_stock_price.invoke(tool_call["args"])
            messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
        
        if tool_call["name"] == "get_exchange_rate":
            rate = get_exchange_rate.invoke(tool_call["args"])
            messages.append(ToolMessage(content=rate,tool_call_id=tool_call["id"]))

    ai_message_2 = llm_with_tools.invoke(messages)

    for tool in ai_message_2.tool_calls:
        if tool["name"] == "calculate_investment":
            investment = calculate_investment.invoke(tool["args"])
            print(investment)
            messages.append(ToolMessage(content=investment,tool_call_id=tool["id"]))

    print(messages)