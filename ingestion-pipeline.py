'''
This is the part one of Retrival Augmented Generation,
This involves:

    1. Loading Documents (in real life this will be actual company docs)

    2. Convert them into Chunks 

    3. Embedd them - uses OPENAI's embedding model

    4. Store the embeddings in a Vectore ems Database

'''

# Dependencies :
#   Langchain , Langchain - community, Langchain - Text Splitting (FOR CHUNKING)
#   Langchain - OpenAI - for embedding model.
#   Langchain - Chroma - for Vector Database


''' The imports are basically for learning, research and find optimum ones for customised performance'''

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader        # This helps openning files like .docx / .pdf /any documents from DIRECTORIES.
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma         # This is for storing vector embeddings in DATABASE.  - This is for local hosting
from dotenv import load_dotenv          # This helps open environment variables --> This will be required by the API's if any are being used (And i will use API)

load_dotenv() # Load the environment files






# Function to load files
def load_documents(docs_path="docs"):
    """Load all text files from the directory path"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",       # only focus on text files for now, not even pdf
        loader_cls=TextLoader  # this loader class is only for text files (langchain one we imported)
    )
    
    documents = loader.load() # this returns LANGCHAIN Documents which has page content attribute and metadata attribute
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents




#Split docs into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=0):       #1000 words per chunk
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(      # this is a very basic text splitting class
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks





# convert into vector embeddings
# since we are using OpenAI we need to pay for it and link the API Key
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(        # this does both embedding and storing
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}    # algorithm to compare with the user's querry and the data in database (VERY IMPORTANT)
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("hello world")


    # 1. loading docs
    documents = load_documents()
    

    # 2. split into chunks
    chunks = split_documents(documents)


    # 3. convert into vector embeddings
    vectorstore = create_vector_store(chunks)




if __name__ == "__main__" :
    main()