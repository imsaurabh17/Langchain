from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Annotated, Literal, List

load_dotenv()

class Entity(BaseModel):
    type: Literal['product','order_id','date','issue','service','location','organization','other']
    value: str

class Pipeline(BaseModel):
    summary: Annotated[str,"Provide the summary of the provided review"]
    sentiment: Literal['Positive',"Negative","Neutral"]
    entities: Annotated[List[Entity],"Provide the entities of the review"]
    problem: Annotated[List[str],"List down all the problems of in the review"]
    solution: Annotated[List[str],"Provide the solutions to address those problems"]
    confidence: float = Field(..., ge=0,le=1)

model = ChatGroq(model="llama-3.3-70b-versatile")

reviews = [
    """""
    I ordered a laptop bag last week and it arrived yesterday, but the zipper is broken. Also, the product color is completely different from what was shown on the website. I tried raising a return request but the app keeps throwing an error. Please fix this, it’s very frustrating.
    """,
    "Worst service ever. My headphones stopped working in 2 days and your support team doesn’t respond.",
    "Hi, I really like your products normally, but this time the delivery of my coffee maker was delayed by 5 days. Not a huge issue but just wanted to let you know.",
    "The shoes fit perfectly and look great, but the sole has started peeling off after just one week. Please help.",
    "Bhai product accha tha but packing bilkul bekaar. Box pura dabba hua tha aur charger missing tha. Return process bhi confusing lag raha hai.",
    "The medication you delivered is expired. This is extremely serious. I need an immediate replacement and investigation.",
    "Delivery late. Not happy.",
    "Mera order aaj aana tha par abhi tak nahi aaya. Tracking link bhi open nahi ho raha. Please check.",
    "Amazing. Truly amazing. I paid full price for a premium watch and you sent me something that looks like it came from a toy shop. Great job.",
    "I received the blender but I’m not sure if it’s working correctly. It makes a loud sound. Could be normal, could be a defect."
]

instructions = """
Instruction:
                       You are an information extraction engine.
Your job is ONLY to extract structured data.
You MUST return a valid JSON object that strictly matches the schema.
Follow these rules with ZERO EXCEPTIONS:

- DO NOT wrap arrays in quotes.
  Example: ✔ ["a", "b"]   ✘ "[\"a\", \"b\"]"

- DO NOT wrap objects in quotes.
  Example: ✔ {"type": "product", "value": "speaker"}  
           ✘ "{\"type\": \"product\", \"value\": \"speaker\"}"

- DO NOT wrap numbers in quotes.
  Example: ✔ 0.45 / 1.0   ✘ "0.45"

- DO NOT output any text outside the JSON.

- For sentiment, use EXACTLY: "Positive", "Negative", or "Neutral".
- For each entity, include BOTH "type" and "value" fields.
- For unknown entity types, use "other".

Return ONLY the JSON that matches the schema.
"""

prompt = PromptTemplate(template="""
Instructions : {instructions}, Input: {review}
""",
input_variables=['instructions','review'])

for review in reviews:
    try:
        prompt_input = prompt.format(instructions=instructions,review=review)
        output = model.with_structured_output(Pipeline)
        result = output.invoke(prompt_input)
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error in processing: {e}")

