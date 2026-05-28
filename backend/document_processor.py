import os
import hashlib
import uuid
from typing import List, Dict
import pdfplumber
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="./research_cache")
doc_collection = chroma_client.get_or_create_collection(
    name="user_documents",
    embedding_function=embedding_fn
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_uploaded_file(file_path: str, original_filename: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are allowed.")

    if not text.strip():
        raise ValueError("No text could be extracted from the file.")

    chunks = text_splitter.split_text(text)
    doc_id = str(uuid.uuid4())
    metadatas = [{"doc_id": doc_id, "source": original_filename, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    doc_collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    return doc_id

def delete_document(doc_id: str):
    results = doc_collection.get(where={"doc_id": doc_id})
    ids_to_delete = results["ids"]
    if ids_to_delete:
        doc_collection.delete(ids=ids_to_delete)

def list_documents() -> List[Dict]:
    all_data = doc_collection.get(include=["metadatas"])
    docs_map = {}
    for meta in all_data["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "source": meta["source"],
                "chunks": 0
            }
        docs_map[doc_id]["chunks"] += 1
    return list(docs_map.values())