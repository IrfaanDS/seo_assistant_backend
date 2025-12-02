import os
import numpy as np
import pickle
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load metadata + vectors
metadata = pickle.load(open("models/metadata.pkl", "rb"))
vectors = np.load("models/vectors.npy")

# Setup ChromaDB

client = chromadb.PersistentClient(path="models/chroma_db")
collection = client.get_or_create_collection(
    name="seo_rag",
    metadata={"hnsw:space": "cosine"}
)

# Ensure DB is populated (only first time)
if collection.count() == 0:
    print("⚠️ ChromaDB empty → Populating...")
    ids = [m["id"] for m in metadata]
    texts = [m["text"] for m in metadata]
    metadatas = [{"source": m["source"], "index": i} for i, m in enumerate(metadata)]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=vectors.tolist()
    )
    print("Inserted", len(ids), "chunks.")


def retrieve(query: str, top_k=5):
    """Retrieve relevant chunks from Chroma."""
    query_vec = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k
    )

    documents = results["documents"][0]
    return "\n\n".join(documents)

def generate_answer(query: str, history: list = None): # <-- MODIFIED: Add history parameter
    """Generate SEO answer using Groq with memory."""
    context = retrieve(query)

    # 1. Construct the history string for the prompt
    history_string = ""
    if history:
        history_string = "Previous Conversation:\n" + "\n".join(
            [f"User: {h['question']}\nAssistant: {h['answer']}" for h in history]
        ) + "\n\n"

    prompt = f"""
You are an **Elite SEO Expert Assistant** for a SaaS platform. Your primary function is to provide highly precise, technically accurate, and actionable SEO advice.

**YOUR PRIMARY INSTRUCTIONS:**
1.  **OUTPUT FORMAT & TONE (CRITICAL):** You MUST adopt the persona of a seasoned, direct SEO consultant. **NEVER** mention the 'context', 'documents', 'knowledge base', 'retrieval', or any similar RAG-related terms in your final response to the user. Respond directly and professionally.
2.  **DOMAIN FOCUS:** You must strictly limit your answers to the domain of Search Engine Optimization, Google ranking systems, Core Web Vitals, and structured data. If a user asks a non-SEO question (e.g., "What is the capital of France?"), you must decline.
3.  **CONCISENESS & PRECISION:** Structure your answer as a brief, authoritative response. **Do not use overly verbose or generic filler phrases.** Use technical SEO terminology where appropriate.
4.  **STRICT CONTEXTUALITY (Internal Rule):** Your answer **MUST** be based exclusively on the provided `Relevant Context`. This is an internal constraint; DO NOT mention this rule to the user.
5.  **HISTORY/TONE:** Reference the `History` (if provided) to maintain continuity, but keep your current response focused on the immediate `Query`.

**--- HISTORY ---**
{history_string}

**--- USER QUERY ---**
{query}

**--- RELEVANT CONTEXT (The only source of truth) ---**
{context}

**--- FALLBACK RULE ---**
If the answer cannot be confidently derived from the 'Relevant Context' alone, your *only* response is:
"The provided documents do not contain the required SEO information to answer this question precisely."
"""
    # 3. Create the messages list for the Groq API call
    # The system instruction is the first message.
    messages = [{"role": "system", "content": prompt}] # Pass the full prompt as a system message.
    
    # 4. Use the updated prompt and call the API
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=350
    )

    return resp.choices[0].message.content