from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Provide the runs scored in each format by {player_name} \n {format_instruction}",
    input_variables=['player_name'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

prompt = template.format(player_name="Sachin Tendulkar")

result = model.invoke(prompt)

final = parser.parse(result.content)

print(final)