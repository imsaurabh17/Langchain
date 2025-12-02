from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class Entity(BaseModel):
    type: Literal["product","order_id", "date", "issue", "service", "location", "organization", "other"]
    value: str


class Output(BaseModel):
    summary: Annotated[str,"Provide the neutral summary of the review describing the user's concern."]
    sentiment: Literal["Positive","Negative","Neutral"]
    issues: Annotated[List[str], "List of all the problems faced by the customer"]
    entities: Annotated[List[Entity], "Extract mentioned entity from the review"]
    action : Annotated[List[str], "List of actions to be taken to address the issue"]
    score : Annotated[float, "The confidence score"] = Field(...,ge=0,le=1)


model = ChatGroq(model="llama-3.3-70b-versatile")

format = model.with_structured_output(Output)

result = format.invoke("""
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


Input:
Hi team,
I don’t know what’s happening with your service these days. I ordered a Bluetooth speaker last week (order ID maybe 67213 or something like that—I don’t remember exactly).

The product arrived 3 days late and the packaging was half torn. The speaker works but the battery drains in less than an hour, which is ridiculous. I tried calling your support number but it keeps saying ‘all agents are busy’ and then disconnects.

Honestly this is the second time I’m facing issues. If this doesn’t get resolved soon, I may just switch to another platform. Please look into this ASAP.
""")

print(result.model_dump_json(indent=4))