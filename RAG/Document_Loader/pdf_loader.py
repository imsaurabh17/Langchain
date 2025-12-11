from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:/Users/RENTIT/Downloads/1500_Suhas_ActualScore_Zulaikha_Saurabh.pdf")

docs = loader.load()

print(docs[0].page_content)