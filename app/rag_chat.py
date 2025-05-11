from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.utils import load_chunks, load_faiss_index
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

class RAGChatBot:
    def __init__(self):
        # Embedder → stays on GPU
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")

        # Load chunks & index…
        self.chunks = load_chunks("index")
        self.index = load_faiss_index("index")

        # TOKENIZER: swap model name here
        MODEL_NAME = "kittn/mistral-7B-v0.1-hf"  # or "tiiuae/falcon-7b-instruct", or "meta-llama/Llama-2-7b-chat-hf" or "kittn/mistral-7B-v0.1-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # MODEL: load in 4-bit or 8-bit for Falcon/Mistral, FP16 for Llama2
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,              # for Mistral/Falcon
            device_map="auto",
            low_cpu_mem_usage=True,
            llm_int8_enable_fp32_cpu_offload=True # for Mistral/Falcon
            # torch_dtype=torch.float16, # for Llama2
            # quant_type="nf4", # for falcon 
        )
        # No manual .to()—accelerate handles placement
        print(f"[DEBUG] Device map: {self.model.hf_device_map}")

    def get_answer(self, query: str, k: int = 3) -> str:
        # Build a custom GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=200,
            do_sample=True,                 # use sampling, not greedy
            top_p=0.92,                     # nucleus sampling
            top_k=50,                       # optional top-k
            temperature=0.85,               # add randomness
            repetition_penalty=1.2,         # penalize repeats
            no_repeat_ngram_size=3          # forbid 3-gram repeats
        )

        query_vec = self.embedder.encode(query, convert_to_numpy=True)
        _, I = self.index.search(query_vec.reshape(1, -1), k)
        context = "\n\n---\n\n".join(self.chunks[i] for i in I[0])

        prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,   # apply the custom config :contentReference[oaicite:7]{index=7}
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("A:")[-1].strip()
