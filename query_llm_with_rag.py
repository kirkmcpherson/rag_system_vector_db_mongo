import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from pymongo import MongoClient

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING')
gemini_api_keys = os.getenv('GEMINI_API_KEYS')

# Initiate required modals.
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_keys)

def convert_text_to_embeddings(data, model):
    # Initiate the embedding model.
    embedding = model.encode(data)
    return embedding.tolist()

def get_mongodb_collection(mongodb_connection_string, db_name, collection_name):
    client = MongoClient(mongodb_connection_string)
    return client[db_name][collection_name]

def get_relevant_chunks(collection, query, chunks_required=3):
  query_embedding = convert_text_to_embeddings(query, model)
  pipeline = [
    {
      "$vectorSearch": {
        "index": "vector_index",
        "queryVector": query_embedding,
        "path": "embedding",
        "exact": True,
        "limit": chunks_required
      }
    },
    {
      "$project": {
        "_id": 0,
        "text": 1
      }
    }
  ]

  results = list(collection.aggregate(pipeline))

  return " ".join([doc["text"] for doc in results])


def generate_response(prompt, relevant_chunk_of_data, llm):
  # Create a prompt template
  prompt_template = PromptTemplate(
    input_variables=["prompt", "relevant_chunk_of_data"], 
    template= """
      Use the following pieces of context to answer the question at the end.

      Context: {relevant_chunk_of_data}

      User's Question: {prompt}
      """
    )

  formatted_prompt = prompt_template.format(
    prompt=prompt,
    relevant_chunk_of_data=relevant_chunk_of_data
  )

  print("FULL FORMATTED PROMPT:")
  print(formatted_prompt)
    
  # Chain the template and instance
  chain = prompt_template | llm

  # Invoke the chain by passing the input variables of prompt
  response = chain.invoke({
    "prompt":prompt,
    "relevant_chunk_of_data": relevant_chunk_of_data
  })

  # Return the response
  return response.content

# Get MongoDB collection.
print("Get Collection: Start")
collection = get_mongodb_collection(mongodb_connection_string, "rag_db", "nike_reports")   
print("Get Collection: Done")

query = "What were Adidas's latest revenues?"

print("Get Relevant Data: Start")
relevant_chunks = get_relevant_chunks(collection, query, 3)
print("Get Relevant Data: Done")

print("Get Response: Start")
generated_response = generate_response(query, relevant_chunks, llm)
print("Get Response: End")

print("RESPONSE:")
print(generated_response)