from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template="Generate a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate explanation of the {joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1,model,parser)

chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model, parser)
})

final_chain = RunnableSequence(joke_chain,chain)

print(final_chain.invoke({'topic':'negative guys'}))