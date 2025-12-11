from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='C:/Users/RENTIT/Documents/Leap',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(docs)