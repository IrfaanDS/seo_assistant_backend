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


def generate_answer(query: str):
    """Generate SEO answer using Groq."""
    context = retrieve(query)

    prompt = f"""
You are an expert SEO assistant.

ANSWER THE USER STRICTLY USING THE CONTEXT.

Query:
{query}

Relevant Context:
{context}

If the answer is not in the context, say: "The provided documents do not contain the required information."
"""

    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350
    )

    return resp.choices[0].message.content
