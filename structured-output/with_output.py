from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

class Review(TypedDict):

    review : Annotated[str,"Write a summary of the review for the customers"]
    sentiment : Annotated[str, "Provide the sentiment of the review in short form i.e. pos for positive, neg for negative"]

model = ChatGroq(model="openai/gpt-oss-20b")

output = model.with_structured_output(Review)

result = output.invoke("""
This was so bad I couldn´t finish it. The actresses are so bad at acting it feels like a bad comedy from minute one. The high rated reviews is obviously from friend/family and is pure BS.
""")

# print(result)
print(result.get('sentiment'))
print(result.get('review'))