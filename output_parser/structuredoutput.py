from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class output(BaseModel):
    fact_1: str
    fact_2:str
    fact_3:str

template = PromptTemplate(
    template = "Provide the explanation of {topic}",
    input_variables=['topic']
)

prompt = template.format(topic="Bermuda Triangle")

model = ChatGroq(model="llama-3.1-8b-instant")

llm = model.with_structured_output(output)

result = llm.invoke(prompt)

final = result.model_json_schema(result)

print(final)
