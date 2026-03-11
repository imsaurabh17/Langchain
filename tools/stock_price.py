import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

@tool
def get_stock_price(stock_symbol: str) -> float:
    """This tool will provide the live price of a stock"""
    ticker = yf.Ticker(stock_symbol)

    current_price = ticker.info.get("currentPrice")

    return current_price


if __name__=="__main__":
    print(get_stock_price.invoke({"stock_symbol":"AAPL"}))