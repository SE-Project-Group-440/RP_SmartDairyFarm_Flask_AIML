import torch
import faiss
import numpy as np
import pickle
from utils.utils import encode_text, load_feeding_json, MiniEmbeddingModel,load_all_knowledge

device = "cuda" if torch.cuda.is_available() else "cpu"

docs = load_all_knowledge("data/knowledge_base")

# Load BPE and model
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
model.load_state_dict(torch.load("models/sinhala_embedding_model.pt", map_location=device))
model.eval()

# Generate embeddings
embeddings = []
with torch.no_grad():
    for doc in docs:
        text = doc.question + " " + doc.answer
        ids = encode_text(text, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
        emb = model(ids)
        embeddings.append(emb.cpu())

embeddings = torch.cat(embeddings, dim=0).numpy()
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

faiss.write_index(index, "vectorstore/knowledge.index")
print(f"✅ FAISS index built with {index.ntotal} documents")
