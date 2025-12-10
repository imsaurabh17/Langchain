from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

prompt1 = PromptTemplate(
    template="provide the detailed explanation of the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Provide the summary of {text}",
    input_variables=['text']
)

parser = StrOutputParser()

explanation_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(explanation_chain,branch_chain)

print(chain.invoke({'topic':'World War I'}))