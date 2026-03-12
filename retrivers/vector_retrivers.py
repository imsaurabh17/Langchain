from langchain_community.vectorstores import Chroma
from  langchain_core.documents import Document
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

db = Chroma.from_documents(
    documents=docs,
    collection_name='my_db',
    embedding=embeddings,
    persist_directory='retrivers'
)

query = 'What is LangGraph?'

retriever = db.as_retriever(search_kwargs={'k':2})

result = retriever.invoke(query)

for doc in result:
    print(doc.page_content)