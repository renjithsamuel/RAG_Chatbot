# Script to build the index
import os
import sys

# Add the parent directory of 'app' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils import load_docs, chunk_texts, save_chunks, save_faiss_index
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.utils import load_docs, chunk_texts, chunk_texts_from_pdf, save_chunks, save_faiss_index

DOCS_DIR = "data/docs"
INDEX_DIR = "index"
CHUNK_SIZE = 500

def main():
    print("🚀 Loading documents...")
    docs = load_docs(DOCS_DIR)
    for fname in os.listdir("data/pdf"):
        if fname.lower().endswith(".pdf"):
            pdf_chunks = chunk_texts_from_pdf(os.path.join("data/pdf", fname), CHUNK_SIZE)
            chunks.extend(pdf_chunks)

    chunks = chunk_texts(docs, CHUNK_SIZE)

    print(f"🔍 Chunked into {len(chunks)} segments. Generating embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    print(f"💾 Saving FAISS index and chunks...")
    save_faiss_index(index, INDEX_DIR)
    save_chunks(chunks, INDEX_DIR)
    print("✅ Done. Index ready to use!")

if __name__ == "__main__":
    main()
