import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import time
import re
import pickle
import numpy as np
import torch
import faiss
import csv
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from scripts.utils import (
    encode_text,
    MiniEmbeddingModel,
    load_all_knowledge
)
from scripts.rag_query import build_prompt

# ======================
# Setup
# ======================
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Load knowledge
# ======================
docs = load_all_knowledge("data/knowledge_base")

with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
model.load_state_dict(
    torch.load("models/sinhala_embedding_model.pt", map_location=device)
)
model.eval()

index = faiss.read_index("vectorstore/knowledge.index")

client = Groq()

# ======================
# Retrieval
# ======================
def get_frozen_context(query, top_k=3):
    q_ids = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model(q_ids).cpu().numpy()

    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    _, indices = index.search(q_emb, top_k)

    return "\n\n".join([docs[i].page_content for i in indices[0]])

# ======================
# Metrics
# ======================
def context_overlap(answer, context):
    a = set(answer.split())
    c = set(context.split())
    return len(a & c) / max(len(a), 1)

def sinhala_ratio(text):
    return len(re.findall(r"[\u0D80-\u0DFF]", text)) / max(len(text), 1)

def answer_length(answer):
    return len(answer.split())

def repetition_ratio(text):
    words = text.split()
    return 1 - (len(set(words)) / max(len(words), 1))

def sentence_diversity(text):
    sents = re.split(r"[.!?]", text)
    sents = [s.strip() for s in sents if s.strip()]
    return len(set(sents)) / max(len(sents), 1)

# ======================
# Models
# ======================
MODELS = {
    "llama_70b": "llama-3.3-70b-versatile",
    "llama_8b":  "llama-3.1-8b-instant",
    "llama_guard": "meta-llama/llama-guard-4-12b",
    "gpt_120b": "openai/gpt-oss-120b",
    "gpt_20b": "openai/gpt-oss-20b"
}

# ======================
# Run Evaluation
# ======================
query = "ශ්‍රී ලංකාවේ බහුලව භාවිත වන ගව ආහාර කුමක්ද?"
context = get_frozen_context(query)
prompt = build_prompt(context, query)

results = []

for name, model_id in MODELS.items():
    start = time.time()

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    latency = round(time.time() - start, 2)
    answer = response.choices[0].message.content.strip()

    result = {
        "model": name,
        "latency_sec": latency,
        "context_overlap": round(context_overlap(answer, context), 3),
        "sinhala_ratio": round(sinhala_ratio(answer), 3),
        "answer_length": answer_length(answer),
        "repetition_ratio": round(repetition_ratio(answer), 3),
        "sentence_diversity": round(sentence_diversity(answer), 3)
    }

    results.append(result)

    print(f"\n🧠 Model: {name}")
    for k, v in result.items():
        if k != "model":
            print(f"{k}: {v}")

# ======================
# Save CSV
# ======================
df = pd.DataFrame(results)
df.to_csv("evaluation_results_metrics.csv", index=False, encoding="utf-8")
print("\n✅ Results saved to evaluation_results_metrics.csv")

# ======================
# Ranking
# ======================
df["latency_rank"] = df["latency_sec"].rank(ascending=True)
df["overlap_rank"] = df["context_overlap"].rank(ascending=False)
df["sinhala_rank"] = df["sinhala_ratio"].rank(ascending=False)

ideal_length = 100
df["length_rank"] = (df["answer_length"] - ideal_length).abs().rank()

df["repetition_rank"] = df["repetition_ratio"].rank()
df["diversity_rank"] = df["sentence_diversity"].rank(ascending=False)

df["total_rank"] = (
    df["latency_rank"]
    + df["overlap_rank"]
    + df["sinhala_rank"]
    + df["length_rank"]
    + df["repetition_rank"]
    + df["diversity_rank"]
)

best_overall = df.loc[df["total_rank"].idxmin(), "model"]

print("\n🌟 OVERALL BEST MODEL:", best_overall)

# ======================
# 📊 VISUALIZATION
# ======================
plt.figure()
plt.bar(df["model"], df["latency_sec"])
plt.title("Model Latency Comparison")
plt.ylabel("Seconds")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(df["model"], df["context_overlap"])
plt.title("Context Overlap Score")
plt.ylabel("Overlap Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(df["model"], df["sinhala_ratio"])
plt.title("Sinhala Language Fluency")
plt.ylabel("Sinhala Character Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(df["model"], df["repetition_ratio"])
plt.title("Repetition Ratio (Lower is Better)")
plt.ylabel("Repetition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(df["model"], df["sentence_diversity"])
plt.title("Sentence Diversity")
plt.ylabel("Diversity Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
