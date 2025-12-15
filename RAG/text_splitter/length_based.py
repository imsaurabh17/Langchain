from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:/Users/RENTIT/Downloads/1500_Suhas_ActualScore_Zulaikha_Saurabh.pdf")
docs = loader.load()

spliter = CharacterTextSplitter(
    separator='',
    chunk_size=100,
    chunk_overlap=0
)

result = spliter.split_documents(docs)

print(result)