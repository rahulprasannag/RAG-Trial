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
from langchain_text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma         # This is for storing vector embeddings in DATABASE.  - This is for local hosting
from dotenv import load_dotenv          # This helps open environment variables --> This will be required by the API's if any are being used (And i will use API)

load_dotenv() # Load the environment files





