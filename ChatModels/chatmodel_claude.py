from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatAnthropic(model = 'claude-sonnet-4-5-20250929')

result = model.invoke("Who is the best cricketer?")

print(result)