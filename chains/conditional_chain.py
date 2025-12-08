from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

class Sentiment(BaseModel):
    sentiment: Annotated[str,"Provide the sentiment either as positive or negative"]

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Sentiment)

prompt = PromptTemplate(
    template="Provide the sentiment of the {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template= "Provide the appropriate and short response for this positive {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Provide the appropriate but short response for this negative {feedback}",
    input_variables=['feedback']
)


conditional_chain = prompt | model | parser2

branch_chain = RunnableBranch(
    (lambda x:x.sentiment.lower()=="positive", prompt2 | model | parser),
    (lambda x:x.sentiment.lower()=='negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

chain = conditional_chain | branch_chain

feedback = """
Response was confusing and lacked clarity
"""

result = chain.invoke({'feedback':feedback})

print(result)