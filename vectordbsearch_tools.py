from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings



load_dotenv()

OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')



class VectorSearchTools_chroma():

  
  def dbsearch(query):
        """
        useful to search vector database and returns most relevant chunks
        """
        # Processing PDF and DOCX files
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(embedding_function= embeddings, persist_directory="./chroma_db")

        
        
        retrieved_docs = db.similarity_search(query, k=4)


        plain_texts = [f"\n{doc.page_content} {doc.metadata}" for doc in retrieved_docs]
        sources = "-"
        # Concatenate plain texts and sources into formatted strings
    
        merged_texts = '\n'.join(plain_texts) + '\n' + '\n'.join(sources)


        return plain_texts


if __name__ == "__main__":  
  docs = VectorSearchTools_chroma.dbsearch("what is this website about?")
  print(docs)
  
  

  