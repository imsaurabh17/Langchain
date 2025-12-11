from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="C:/Users/RENTIT/Downloads/transcription_results_enhanced (3).csv",encoding='utf-8')

for row in loader.lazy_load():
    print(row.page_content)