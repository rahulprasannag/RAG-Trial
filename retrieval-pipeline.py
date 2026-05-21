'''
This is the part two of Retrival Augmented Generation,
This involves:

    1. Reciving user's query/prompt

    2. Convert them into Chunks 

    3. Retriving similar chunks in vector database (along with the original english chunks) - IMPORTANT
        i. uses cosine similarity algorithm to find similar chunks

'''

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")      # same model used in ingestion pipeline (otherwise it will not understand - this is similar to encryption)


# pointing the vector database with embedded chunks
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"


#this is the actual retriver (who finds similar chunks)
retriever = db.as_retriever(search_kwargs={"k": 3})     # k is how many top chunks you want - here we want 5 top similar chunks to be retrived.
#we can also specify the score threshold like this:
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")