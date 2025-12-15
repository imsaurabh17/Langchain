from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="AB Devilliers is the best.",
        metadata = {'Team':"RCB"}
    ),
    Document(
        page_content="Cristiano Ronaldo is Goat",
        metadata={'Team':"Real Madrid"}
    ),
    Document(
        page_content="Virat kohli is a great player",
        metadata={'Team':'RCB'}
    ),
    Document(
        page_content="What you think you become",
        metadata={'Content':'Motivational'}
    ),
    Document(
        page_content="Work on yourself",
        metadata={'content':'Motivational'}
    )
]


db = Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    persist_directory='./chroma_db'
)

db.persist()