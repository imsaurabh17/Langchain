import os
import json
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

EXCHANGE_RATE_API = os.getenv("EXCHANGE_RATE_API")

messages = []

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
def convert_value(source_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    This tool will convert the source value into the target value with the help of conversion rate.
    """

    result = source_value * conversion_rate

    return result

tools = [get_convert_rate, convert_value]

llm = ChatGroq(model="llama-3.3-70b-versatile")

llm_with_tools = llm.bind_tools(tools)


if __name__=="__main__":

    query = HumanMessage("convert 250 AUD to USD")

    messages.append(query)

    ai_message = llm_with_tools.invoke(messages)

    messages.append(ai_message)

    for tool in ai_message.tool_calls:
        if tool["name"] == "get_convert_rate":
            output = get_convert_rate.invoke(tool["args"])
            messages.append(ToolMessage(content=output,tool_call_id=tool["id"]))
            
        if tool["name"] == "convert_value":
            args = {**tool["args"],"conversion_rate":output}
            output = convert_value.invoke(args)
            messages.append(ToolMessage(content=output,tool_call_id=tool["id"]))

    result = llm_with_tools.invoke(messages)
    print(result.content)