from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

class person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18,description="Age of the person")
    city: str = Field(description="City of the person where he/she lives")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template="Provide the name, age and city of a fictional {nationality} charcter \n {format_instruction}",
    input_variables=['nationality'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain = template |  model | parser

result = chain.invoke({"nationality":"American"})

print(result)