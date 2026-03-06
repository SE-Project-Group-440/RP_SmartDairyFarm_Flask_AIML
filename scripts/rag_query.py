import os
import torch
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
from scripts.utils import load_feeding_json, MiniEmbeddingModel, encode_text,load_all_knowledge
from scripts.retriever import retrieve_faiss_custom
import re

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

docs = load_all_knowledge("data/knowledge_base")

# Load BPE
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

# Load model
model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
model.load_state_dict(torch.load("models/sinhala_embedding_model.pt", map_location=device))
model.eval()

# Load FAISS
index = faiss.read_index("vectorstore/knowledge.index")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(context: str, query: str) -> str:
    return f"""
ඔබ ශ්‍රී ලංකාවේ කිරි ගව පාලකයින්ට උපදෙස් ලබාදෙන කෘෂිකාර්මික උපදේශකයෙකි.

දැනුම් පදනම:
{context}

ප්‍රශ්නය:
{query}

පිළිතුර:
1. එකම අදහස නැවත නැවත නොලියන්න
2. සරල සිංහල භාෂාව භාවිතා කරන්න
3. අවශ්‍ය නම් උදාහරණ දෙන්න
4. ගව පාලකයාට ප්‍රායෝගික උපදෙස් ලබාදෙන්න
5. පිළිතුර අවසානයේ සම්පූර්ණ වාක්‍යයකින් අවසන් කරන්න
6. එකම වාක්‍ය ආරම්භය නැවත නැවත භාවිතා නොකරන්න
7. Markdown (** , ## , - ) භාවිතා නොකරන්න
"""
def remove_repeated_phrases(text):
    words = text.split()
    cleaned = []
    
    for w in words:
        if len(cleaned) > 5 and w == cleaned[-1]:
            continue
        cleaned.append(w)

    return " ".join(cleaned)


def deduplicate_sentences(text: str):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    seen = set()
    unique = []

    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique.append(s_clean)

    return " ".join(unique)

def rag_answer(query: str, top_k: int = 10):
    # Retrieve
    model.eval()
    q_ids = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model(q_ids).cpu().numpy()
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    distances, indices = index.search(q_emb, top_k)
    retrieved_docs = [docs[i] for i in indices[0]]
    
    if not retrieved_docs:
        return "මෙම ප්‍රශ්නයට සම්බන්ධ තොරතුරු දැනුම් පදනමේ නොමැත."
    
    unique_chunks = list(
        dict.fromkeys(
            [doc.page_content.strip() for doc in retrieved_docs]
        )
    )

    context = "\n\n".join(unique_chunks)

    prompt = build_prompt(context, query)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048
    )
    answer = response.choices[0].message.content.strip()
    answer = deduplicate_sentences(answer)
    answer = remove_repeated_phrases(answer)
    return answer
