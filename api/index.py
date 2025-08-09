import pickle
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# تحميل البيانات المحفوظة
with open("api/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index("api/faiss_index.bin")

model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    query: str

def search_faiss(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [chunks[i] for i in indices[0]]
    return results

@app.post("/query")
def query_mikrotik(request: QueryRequest):
    question = request.query
    results = search_faiss(question, model, index, chunks, top_k=3)
    return {"results": results}
