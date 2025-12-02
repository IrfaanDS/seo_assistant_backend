from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_answer

app = FastAPI(title="SEO Assistant RAG API")

# --- NEW: In-memory storage for session history ---
# Key: session_id (str)
# Value: list of {"question": str, "answer": str}
conversation_history = {}
# --- END NEW ---

class Query(BaseModel):
    session_id: str # <-- MODIFIED: Add session_id
    question: str

@app.get("/")
def root():
    return {"message": "SEO Assistant RAG API running"}

@app.post("/ask")
def ask_question(payload: Query):
    session_id = payload.session_id
    question = payload.question

    # --- NEW: Retrieve history ---
    # Get or initialize history for the session
    current_history = conversation_history.get(session_id, [])
    # --- END NEW ---

    # --- MODIFIED: Pass history to generate_answer ---
    answer = generate_answer(question, history=current_history)
    
    # --- NEW: Update history ---
    # Append the new turn to the history
    current_history.append({"question": question, "answer": answer})
    conversation_history[session_id] = current_history
    # --- END NEW ---

    return {"answer": answer}
