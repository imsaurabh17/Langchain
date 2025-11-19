from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "Hi, how are you?",
    "I am fine",
    "What about you?"
]

vector = embedding.embed_query("Who is ronaldo?")

vectors = embedding.embed_documents(docs)

print(vectors)