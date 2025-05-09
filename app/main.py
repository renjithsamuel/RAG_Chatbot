from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chat import RAGChatBot

app = FastAPI()
chatbot = RAGChatBot()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    answer = chatbot.get_answer(query.question)
    return {"answer": answer}
