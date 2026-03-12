from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content='Langchain is the Best Framework to build LLM apps'),
    Document(page_content='Chroma is a Vector Database'),
    Document(page_content='Langgraph is the best tool to build Agentic AI apps'),
    Document(page_content='AI is the future'),
    Document(page_content='Transformers revolutionised AI field')
]

db = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = db.as_retriever(search_type='mmr',search_kwargs={'k':3,'lambda_mult':1})

query = 'What is LangGraph?'

result = retriever.invoke(query)

for doc in result:
    print(doc.page_content)