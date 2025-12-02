from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system","You are a helpful agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",'{query}')
])

chat_history = []
with open("chats.txt","r") as f:
    chat_history.append(f.readlines())

prompt = chat_template.invoke({"chat_history":chat_history,"query":"What's the status?"})

print(prompt)