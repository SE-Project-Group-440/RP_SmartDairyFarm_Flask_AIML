from scripts.utils import load_all_knowledge, pickle, os
from scripts.nlu import train_nlu

docs = load_all_knowledge("data/knowledge_base")

# Load BPE
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

train_nlu(docs, bpe_merges, bpe_vocab, epochs=20)
