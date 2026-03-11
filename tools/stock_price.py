import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
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


if __name__=="__main__":
    print(get_stock_price.invoke({"stock_symbol":"AAPL"}))
    print(get_exchange_rate.invoke({"source_currency":"USD","target_currency":"INR"}))