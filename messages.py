from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage(content="You are a helpful assitant"),
    HumanMessage(content="Which one is easier to learn? Llmmaindex or langchain and which on in more demand?")
]

model = ChatGroq(model="openai/gpt-oss-120b")

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)