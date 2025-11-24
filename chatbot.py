from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

chat_history = []

while True:
    user = input("You: ")
    chat_history.append(user)
    if user == "exit":
        break
    result = model.invoke(user)
    chat_history.append(result.content)
    print(result.content)