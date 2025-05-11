import pickle
import faiss
from fastapi import FastAPI, File, UploadFile
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app.rag_chat import RAGChatBot

import torch
import logging

from app.utils import chunk_texts_from_pdf, load_chunks, load_faiss_index

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU check
if torch.cuda.is_available():
    logger.info(f"🔥 CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("⚠️ CUDA is NOT available! Running on CPU. Hug your GPU.")


app = FastAPI()
chatbot = RAGChatBot()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    answer = chatbot.get_answer(query.question)
    return {"answer": answer}

# Preload embedder & index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = load_faiss_index("index")
chunks = load_chunks("index")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # 1. Save upload
    ext = file.filename.split(".")[-1].lower()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 2. Chunk
    if ext == "pdf":
        new_chunks = chunk_texts_from_pdf(temp_path)
    else:
        text = open(temp_path, "r", encoding="utf-8").read()
        new_chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # 3. Embed & add to FAISS
    embeddings = embedder.encode(new_chunks, convert_to_numpy=True)
    index.add(np.array(embeddings))

    # 4. Update metadata & persist
    chunks.extend(new_chunks)
    with open("index/texts.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "index/faiss_index.bin")

    return {"ingested_chunks": len(new_chunks)}
