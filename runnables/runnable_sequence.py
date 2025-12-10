from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    template="Tell a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the intution behind the {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({"topic":"Data Science"}))