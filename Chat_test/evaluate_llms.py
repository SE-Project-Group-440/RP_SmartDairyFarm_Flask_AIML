import time
import re
import pickle
import numpy as np
import torch
import faiss
import csv
import matplotlib.pyplot as plt
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

# Load BPE
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

# Load embedding model
model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
model.load_state_dict(
    torch.load("models/sinhala_embedding_model.pt", map_location=device)
)
model.eval()

# Load FAISS index
index = faiss.read_index("vectorstore/knowledge.index")

# Groq client
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
# Evaluation metrics
# ======================
def context_overlap(answer, context):
    a = set(answer.split())
    c = set(context.split())
    return len(a & c) / max(len(a), 1)

def sinhala_ratio(text):
    return len(re.findall(r"[\u0D80-\u0DFF]", text)) / max(len(text), 1)

def answer_length(answer):
    return len(answer.split())

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
        "answer_length": answer_length(answer)
    }

    results.append(result)

    print(f"\n🧠 Model: {name}")
    for k, v in result.items():
        if k != "model":
            print(f"{k}: {v}")

# ======================
# Save CSV
# ======================
with open("evaluation_results_metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "latency_sec",
            "context_overlap",
            "sinhala_ratio",
            "answer_length"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n✅ Evaluation results saved to evaluation_results_metrics.csv")

# ======================
# Plotting
# ======================
models = [r["model"] for r in results]

def plot_bar(values, title, ylabel):
    plt.figure(figsize=(6,4))
    plt.bar(models, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Models")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, str(v), ha="center", fontweight="bold")
    plt.show()

plot_bar(
    [r["latency_sec"] for r in results],
    "Latency Comparison",
    "Seconds"
)

plot_bar(
    [r["context_overlap"] for r in results],
    "Context Overlap",
    "Overlap Ratio"
)

plot_bar(
    [r["sinhala_ratio"] for r in results],
    "Sinhala Language Fluency",
    "Sinhala Character Ratio"
)

plot_bar(
    [r["answer_length"] for r in results],
    "Answer Length",
    "Word Count"
)

# ======================
# Determine overall best model
# ======================

# Step 1: Rank each model per metric
# Lower latency is better → rank ascending
# Other metrics higher is better → rank descending

import pandas as pd

df = pd.DataFrame(results)

# Rank metrics
df["latency_rank"] = df["latency_sec"].rank(ascending=True)
df["overlap_rank"] = df["context_overlap"].rank(ascending=False)
df["sinhala_rank"] = df["sinhala_ratio"].rank(ascending=False)
# For answer length, closer to ideal length is better
ideal_length = 100
df["length_rank"] = (df["answer_length"] - ideal_length).abs().rank(ascending=True)

# Step 2: Sum ranks for overall performance
df["total_rank"] = df["latency_rank"] + df["overlap_rank"] + df["sinhala_rank"] + df["length_rank"]


# ======================
# Determine best models per metric
# ======================

best_latency = df.loc[df["latency_sec"].idxmin(), "model"]
best_overlap = df.loc[df["context_overlap"].idxmax(), "model"]
best_sinhala = df.loc[df["sinhala_ratio"].idxmax(), "model"]
ideal_length = 100
best_length = df.loc[(df["answer_length"] - ideal_length).abs().idxmin(), "model"]

# Step 2: Best overall model → lowest total rank
best_overall = df.loc[df["total_rank"].idxmin(), "model"]

# Print results
print("\n🏆 BEST MODELS PER METRIC")
print(f"Fastest (lowest latency): {best_latency}")
print(f"Highest context overlap: {best_overlap}")
print(f"Best Sinhala fluency: {best_sinhala}")
print(f"Best answer length (closest to {ideal_length} words): {best_length}")

print(f"\n🌟 OVERALL BEST MODEL: {best_overall}")

