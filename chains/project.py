import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated, Dict
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

class Error(BaseModel):
    error: Annotated[Dict[str,str],"Provide the each error in the dict like error1:corresponding error etc."]

parser = PydanticOutputParser(pydantic_object=Error)

prompt = PromptTemplate(
    template="Find the error in the code and explain exactly where's the error from {code}\n {format_instruction}",
    input_variables=['code'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain = prompt | model | parser

code = """"
import json

def calculate_discount(price, discount):
    if price < 0 or discount < 0:
        return "Invalid input"
    final_price = price - price * discount/100
    if final_price > price:
        print("Discount applied incorrectly")
    return final_price

def load_user_data(file_path):
    with open("file_path", "r") as f:   # 
        data = json.load(f)
    return data

def find_max_value(numbers):
    max_val = 0               # 
    for n in numbers:
        if n > max_val:
            max_val = n
    return max_val

def divide(a, b):
    return a / b              # 

def get_user_age(user):
    return user["age"]        # 

# Main workflow
def main():
    prices = [100, 200, -50, 300]   # 
    discounts = [10, 20, 150]       # 

    for i in range(len(prices)):
        print("Final price:", calculate_discount(prices[i], discounts[i]))

    nums = [-5, -10, -3]            # 
    print("Max:", find_max_value(nums))

    print("Division:", divide(10, 0))   # 

    user = {"name": "Saurabh"}          # 
    print("Age:", get_user_age(user))

    data = load_user_data("users.json") # 
    print("Loaded:", data)

main()
"""

result = chain.invoke({'code':code})

errors = json.dumps(result.error,indent=4)

print(errors)