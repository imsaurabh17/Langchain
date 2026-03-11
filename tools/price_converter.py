import os
import json
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests

load_dotenv()

EXCHANGE_RATE_API = os.getenv("EXCHANGE_RATE_API")

@tool
def get_convert_rate(source_currency: str, target_currency: str) -> int:
    """
    This function will take source currency and the target currency and provide the live conversion rate of it using exchange rate api key.
    """
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API}/pair/{source_currency}/{target_currency}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        conversion_rate = data['conversion_rate']
        return conversion_rate
    else:
        return f"Issue"
    

@tool    
def convert_value(source_value: float, conversion_rate: float) -> float:
    """
    This tool will convert the source value into the target value with the help of conversion rate.
    """

    result = source_value * conversion_rate

    return result


if __name__=="__main__":

    conversion_rate = get_convert_rate.invoke({"source_currency": "AED", "target_currency": "INR"})

    print(convert_value.invoke({"source_value":10,"conversion_rate":conversion_rate}))