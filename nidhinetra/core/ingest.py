import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

def ingest_pdf_to_chroma(pdf_path: str, namespace: str = "default", user_id: str = "anonymous", year: str = "2024") -> int:
    documents = PyPDFLoader(pdf_path).load()
    for doc in documents:
        doc.metadata.update({"user_id": user_id, "year": year})

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

    db.add_documents(documents, ids=[str(i) for i in range(len(documents))])
    return len(documents)


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
    #results = db.similarity_search_with_score(query, k=k, where=filters["$and"])

    return results
