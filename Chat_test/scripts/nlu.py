#
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scripts.utils import encode_text, load_all_knowledge
import pickle
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset for intent classification
class IntentDataset(Dataset):
    def __init__(self, docs, bpe_merges, bpe_vocab, intent_map):
        self.docs = docs
        self.bpe_merges = bpe_merges
        self.bpe_vocab = bpe_vocab
        self.intent_map = intent_map

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        x = encode_text(doc.question + " " + doc.answer, self.bpe_merges, self.bpe_vocab)
        y = self.intent_map.get(doc.metadata.get("category", "feeding"), 0)
        return x, torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs, ys

# Intent classifier
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(emb_dim*2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(-1)
        return self.classifier(x)

# Train NLU
def train_nlu(docs, bpe_merges, bpe_vocab, num_classes=5, epochs=20):
    intent_map = {cat:i for i, cat in enumerate(["feeding","breeding","milk_production","disease","management"])}
    dataset = IntentDataset(docs, bpe_merges, bpe_vocab, intent_map)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = IntentClassifier(len(bpe_vocab), num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} / {epochs}, Loss: {total_loss/len(loader):.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sinhala_nlu_classifier.pt")
    print("NLU model saved.")
    return model, intent_map

# Load trained NLU
def load_nlu(bpe_merges, bpe_vocab):
    from scripts.nlu import IntentClassifier
    model = IntentClassifier(len(bpe_vocab))
    model.load_state_dict(torch.load("models/sinhala_nlu_classifier.pt", map_location=device))
    model.to(device)
    model.eval()
    intent_map = {0:"feeding", 1:"breeding", 2:"milk_production", 3:"disease", 4:"management"}
    return model, intent_map

# Predict intent
def classify_intent(query, model, bpe_merges, bpe_vocab, intent_map):
    model.eval()
    x = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return intent_map[pred]
