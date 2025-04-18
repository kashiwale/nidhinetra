from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Ingest PDF into Chroma vector DB
def ingest_pdf_to_chroma(pdf_path: str, namespace: str, user_id: str, year: str) -> int:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Add metadata to each chunk
    for doc in documents:
        doc.metadata["user_id"] = user_id
        doc.metadata["year"] = year

    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./chroma_store",
        collection_name=namespace,
        embedding_function=embeddings
    )
    db.add_documents(documents)
    return len(documents)


def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])


# Search documents in Chroma vector DB
def search_documents(query: str, namespace: str, user_id: str, year: str, k: int = 5):
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

    filters = {
        "$and": [
            {"user_id": {"$eq": user_id}},
            {"year": {"$eq": year}}
        ]
    }

    results = db.similarity_search_with_score(query, k=k, filter=filters)

    return results

    return results