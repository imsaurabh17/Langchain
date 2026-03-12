from langchain_community.retrievers import WikipediaRetriever

retriver = WikipediaRetriever(top_k_results=2,lang='en')

query = 'Ronaldo vs Messi'

docs = retriver.invoke(query)

for i,doc in enumerate(docs):
    print(f"Result -> {i+1}")
    print(doc.page_content)