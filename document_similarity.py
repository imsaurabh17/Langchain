from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "Virat kohli is a great batsman",
    "Jasprit Bumrah is very good bowler",
    "My inspiration is Cristiano Ronaldo",
    "The Goat of football is Cristiano"
]

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

question = input("Enter your question: ")

vectors = embedding.embed_documents(docs)

vector = embedding.embed_query(question)

result = cosine_similarity(vectors,[vector])

similarity = max(enumerate(result),key= lambda x:x[1])

index = similarity[0]

print(docs[index])
