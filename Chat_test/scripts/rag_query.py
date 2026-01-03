import os
import torch
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
from scripts.utils import load_feeding_json, MiniEmbeddingModel, encode_text,load_all_knowledge
from scripts.retriever import retrieve_faiss_custom

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

නීති (අත්‍යවශ්‍යයි):
- එකම අදහස නැවත නැවත නොලියන්න
- එකම වාක්‍ය රටාව නැවත භාවිත නොකරන්න
- ලැයිස්තු අයිතම අතර අදහස් පුනරාවර්තනය නොවිය යුතුය
දැනුම් පදනමෙන් පිටත තොරතුරු එකතු නොකරන්න.

දැනුම් පදනම:
{context}

ප්‍රශ්නය:
{query}

පිළිතුර:
- ගව පාලකයාට තේරුම් ගත හැකි ලෙස
- අවශ්‍ය නම් උදාහරණ සහ හේතු සමඟ
- ප්‍රායෝගික උපදෙස් ලෙස
- සියලුම තොරතුරු සම්පූර්ණයෙන් සපයන්න, වාක්‍ය හෝ අදහස් අඩංගු නොවී අඩක් නොනවත්වා.
- 
"""

def rag_answer(query: str, top_k: int = 3):
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
    
     # ✅✅ ADD THIS BLOCK (DEDUPLICATION FIX) ✅✅
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
    return response.choices[0].message.content.strip()
