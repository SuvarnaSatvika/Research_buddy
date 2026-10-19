import os

import fitz
import re
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore

def extract_pdf_data(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text") + "\n"
    # Regex for finding year
    year_match = re.search(r'\b(20[0-2][0-9])\b', full_text[:2000])
    publication_year = year_match.group(1) if year_match else "Unknown"

    # Extract file name from path and remove extension to use it as Title
    title = file_path.split("/")[-1].replace(".pdf", "")

    return full_text, {"title" : title, "year" : publication_year}

def build_vector_database(file_path):
    print (f"Processing: {file_path}")

    text, metadata = extract_pdf_data(file_path)

    # Use extracted metadata to create Langchain document
    docs = [Document(page_content = text, metadata = metadata)]
    # Nomic embedding model via Ollama
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    # Set up ChromaDB
    vectorstore = Chroma(
        collection_name = "research_papers",
        embedding_function = embeddings,
        persist_directory = "./chroma_db"
    )

    # Initialize storage location
    store = InMemoryStore()

    # Parent chunk for better context
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    # Child chunk for search target
    child_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 50)

    retriever = ParentDocumentRetriever(
        vectorstore = vectorstore,
        docstore = store,
        child_splitter = child_splitter,
        parent_splitter = parent_splitter,
    )
    print("Chunking document and generating embeddings...This may take a moment")
    retriever.add_documents(docs, id = None)
    print ("Vectors stored in chroma_db successfully")
    return retriever

if __name__ == "__main__":
    folder_path = "./data"

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            build_vector_database(file_path)
    print ("Database updated")