import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING')

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def get_data_from_pdf(pdf_url):
  loader = PyPDFLoader(pdf_url)
  return loader.load()

def split_text_into_chunks(data, chunk_size=400, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  return text_splitter.split_documents(data)

def convert_text_to_embeddings(data, model):
    # Initiate the embedding model.
    embedding = model.encode(data)
    return embedding.tolist()

def get_mongodb_collection(mongodb_connection_string, db_name, collection_name):
    client = MongoClient(mongodb_connection_string)
    return client[db_name][collection_name]

def create_search_index(collection, index_name):
   # Create the search index
  search_index_model = SearchIndexModel(
    definition = {
      "fields": [
        {
          "type": "vector",
          "numDimensions": 768,
          "path": "embedding",
          "similarity": "cosine"
        }
      ]
    },
    name = index_name,
    type = "vectorSearch"
  )

  collection.create_search_index(model=search_index_model)


print("Fetch Data: Start")
#data = get_data_from_pdf("https://s1.q4cdn.com/806093406/files/doc_financials/2025/q4/Q4-FY25_Press-Release_FINAL.pdf")
print("Fetch Data: Complete")

print("Text Chunking: Start")
#documents = split_text_into_chunks(data, 400, 20)
print("Text Chunking: Done")

#docs_to_insert = [{
#    "text": doc.page_content,
#    "embedding": convert_text_to_embeddings(doc.page_content, model)
#} for doc in documents]

print("Get Collection: Start")
collection = get_mongodb_collection(mongodb_connection_string, "rag_db", "nike_reports")   
print("Get Collection: Done")

print("Insert Documents: Start")
#collection.insert_many(docs_to_insert)
print("Insert Documents: Done")

print("Create Index: Start")
create_search_index(collection, "vector_index")
print("Create Index: Done")