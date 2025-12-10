from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Create a joke on {topic}",
    input_variables=['topic']
)

def word_counter(joke):
    return len(joke.split())


joke_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

final = RunnableSequence(joke_chain,parallel_chain)

result = final.invoke({'topic':'Football'})

final_result = f"Joke - {result['joke']} \n Word_count - {result['word_count']}"

print(final_result)