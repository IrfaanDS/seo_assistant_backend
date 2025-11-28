from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_answer

app = FastAPI(title="SEO Assistant RAG API")

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "SEO Assistant RAG API running"}

@app.post("/ask")
def ask_question(payload: Query):
    answer = generate_answer(payload.question)
    return {"answer": answer}
