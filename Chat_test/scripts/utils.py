import json
import pickle
import os
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class KnowledgeDocument:
    def __init__(self, question: str, answer: str, metadata: Dict):
        self.question = question
        self.answer = answer
        self.page_content = f"ප්‍රශ්නය: {question}\nපිළිතුර: {answer}"
        self.metadata = metadata


def load_feeding_json(json_path: str) -> List[KnowledgeDocument]:
    documents = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        documents.append(
            KnowledgeDocument(
                question=entry["question"],
                answer=entry["answer"],
                metadata={
                    "id": entry.get("id"),
                    "category": entry.get("category", "feeding"),
                    "source": "feeding.json",
                    "language": "si"
                }
            )
        )
    return documents

# BPE Tokenizer
def prepare_corpus(docs):
    texts = []
    for doc in docs:
        texts.append(doc.question.strip())
        texts.append(doc.answer.strip())
    return texts


def init_bpe_vocab(corpus):
    vocab = defaultdict(int)
    for text in corpus:
        chars = " ".join(list(text)) + " </w>"
        vocab[chars] += 1
    return vocab


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab


def encode_text(text, bpe_merges, bpe_vocab):
    tokens = list(text)
    for a, b in bpe_merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i + 1] == b:
                tokens[i:i + 2] = [a + b]
            else:
                i += 1
    return torch.tensor(
        [bpe_vocab.get(tok, bpe_vocab["<unk>"]) for tok in tokens],
        dtype=torch.long
    )



# Dataset
class QAEmbeddingDataset(Dataset):
    def __init__(self, docs, bpe_merges, bpe_vocab):
        self.docs = docs
        self.bpe_merges = bpe_merges
        self.bpe_vocab = bpe_vocab

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        q = encode_text(self.docs[idx].question, self.bpe_merges, self.bpe_vocab)
        a = encode_text(self.docs[idx].answer, self.bpe_merges, self.bpe_vocab)
        return q, a


def collate_fn(batch):
    qs, as_ = zip(*batch)
    qs = pad_sequence(qs, batch_first=True)
    as_ = pad_sequence(as_, batch_first=True)
    return qs, as_



# Embedding Model

class MiniEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            emb_dim, emb_dim, batch_first=True, bidirectional=True
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(-1)
        return x

def load_qa_json(json_path: str) -> List[KnowledgeDocument]:
    documents = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_name = os.path.basename(json_path)

    for entry in data:
        documents.append(
            KnowledgeDocument(
                question=entry["question"],
                answer=entry["answer"],
                metadata={
                    "id": entry.get("id"),
                    "category": entry.get("category", source_name.replace(".json", "")),
                    "source": source_name,
                    "language": "si"
                }
            )
        )

    print(f"Loaded {len(documents)} docs from {source_name}")
    return documents

def load_all_knowledge(base_dir: str) -> List[KnowledgeDocument]:
    all_docs = []

    for file in os.listdir(base_dir):
        if file.endswith(".json"):
            json_path = os.path.join(base_dir, file)
            docs = load_qa_json(json_path)
            all_docs.extend(docs)

    print(f"Total knowledge documents loaded: {len(all_docs)}")
    return all_docs
