import os
import pickle
from typing import List
import faiss

def load_docs(path: str) -> List[str]:
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def chunk_texts(docs: List[str], chunk_size: int = 500) -> List[str]:
    chunks = []
    for doc in docs:
        chunks.extend([doc[i:i + chunk_size] for i in range(0, len(doc), chunk_size)])
    return chunks

def save_chunks(chunks: List[str], index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "texts.pkl"), "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(index_dir: str) -> List[str]:
    with open(os.path.join(index_dir, "texts.pkl"), "rb") as f:
        return pickle.load(f)

def save_faiss_index(index, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss_index.bin"))

def load_faiss_index(index_dir: str):
    return faiss.read_index(os.path.join(index_dir, "faiss_index.bin"))
