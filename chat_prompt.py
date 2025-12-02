from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

prompt = ChatPromptTemplate([
    ("system","You are a helpful {domain} expert"),
    ("human","Explain about {topic}")
])

prompt = prompt.invoke({"domain":"Football","topic":"Offside Rule"})

print(prompt)