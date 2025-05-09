from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.utils import load_chunks, load_faiss_index
import torch

class RAGChatBot:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = load_chunks("index")
        self.index = load_faiss_index("index")

        self.tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mosaicml/mpt-7b-instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def get_answer(self, query: str, k: int = 3) -> str:
        query_vec = self.embedder.encode(query, convert_to_numpy=True)
        _, I = self.index.search(query_vec.reshape(1, -1), k)
        context = "\n\n---\n\n".join(self.chunks[i] for i in I[0])

        prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("A:")[-1].strip()
