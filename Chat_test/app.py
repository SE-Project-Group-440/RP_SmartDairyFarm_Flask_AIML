from fastapi import FastAPI
from pydantic import BaseModel
from scripts.rag_query import rag_answer

app = FastAPI(title="Sinhala Dairy RAG API")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = rag_answer(req.query)   # ✅ FIX HERE
    return {"answer": answer}
