# scripts/retriever.py
import pickle
import faiss
import numpy as np
import torch
from pathlib import Path

from utils.utils import encode_text, load_feeding_json,MiniEmbeddingModel,load_all_knowledge


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
VECTOR_DIR = BASE_DIR / "vectorstore"
DATA_DIR = BASE_DIR / "data"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
with open(MODEL_DIR / "bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)

with open(MODEL_DIR / "bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

# Load model
vocab_size = len(bpe_vocab)
model = MiniEmbeddingModel(vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_DIR / "sinhala_embedding_model.pt", map_location=device))
model.eval()

# Load FAISS index
index = faiss.read_index(str(VECTOR_DIR / "knowledge.index"))

# Load documents
docs = load_all_knowledge("data/knowledge_base")


def retrieve_faiss_custom(query: str, top_k: int = 10):
    model.eval()
    q_ids = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model(q_ids).cpu().numpy()
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        doc = docs[idx]
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "distance": distances[0][i]
        })
    return results

