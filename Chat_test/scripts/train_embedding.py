import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import pickle
from scripts.utils import QAEmbeddingDataset, collate_fn, encode_text, load_feeding_json, MiniEmbeddingModel,load_all_knowledge

device = "cuda" if torch.cuda.is_available() else "cpu"

docs = load_all_knowledge("data/knowledge_base")

# Load BPE
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

dataset = QAEmbeddingDataset(
    docs,
    bpe_merges=bpe_merges,
    bpe_vocab=bpe_vocab
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    total_loss = 0
    for q, a in dataloader:
        q, a = q.to(device), a.to(device)
        q_emb, a_emb = model(q), model(a)
        scores = F.cosine_similarity(q_emb.unsqueeze(1), a_emb.unsqueeze(0), dim=-1)
        labels = torch.arange(q.size(0)).to(device)
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(30):
    loss = train_epoch(model, dataloader, optimizer, device)
    print(f"Epoch {epoch+1} Loss: {loss:.4f}")

torch.save(model.state_dict(), "models/sinhala_embedding_model.pt")
print("Embedding model saved.")
