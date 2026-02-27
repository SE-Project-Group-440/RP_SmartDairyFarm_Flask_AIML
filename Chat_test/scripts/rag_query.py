import os
import torch
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv

from google import genai
from google.genai import types # ✅ Latest Gemini SDK

from scripts.utils import load_all_knowledge, encode_text, MiniEmbeddingModel
from scripts.nlu import load_nlu, classify_intent

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure GOOGLE_API_KEY is set in environment
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"  # optional if not already set

# Make sure GOOGLE_API_KEY is in your .env file
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------------------------------------------------
# Load knowledge base
# ------------------------------------------------------------------
docs = load_all_knowledge("data/knowledge_base")

# ------------------------------------------------------------------
# Load BPE tokenizer
# ------------------------------------------------------------------
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)

with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

# ------------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------------
model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
model.load_state_dict(
    torch.load("models/sinhala_embedding_model.pt", map_location=device)
)
model.eval()

# ------------------------------------------------------------------
# Load NLU model
# ------------------------------------------------------------------
nlu_model, intent_map = load_nlu(bpe_merges, bpe_vocab)

# ------------------------------------------------------------------
# Prompt Builder
# ------------------------------------------------------------------
def build_prompt(context: str, query: str) -> str:
    return f"""
ඔබ ශ්‍රී ලංකාවේ කිරි ගව පාලකයින්ට උපදෙස් ලබාදෙන කෘෂිකාර්මික උපදේශකයෙකි.

නීති (අත්‍යවශ්‍යයි):
- එකම අදහස නැවත නැවත නොලියන්න
- එකම වාක්‍ය රටාව නැවත භාවිත නොකරන්න
- ලැයිස්තු අයිතම අතර අදහස් පුනරාවර්තනය නොවිය යුතුය
- දැනුම් පදනමෙන් පිටත තොරතුරු එකතු නොකරන්න

දැනුම් පදනම:
{context}

ප්‍රශ්නය:
{query}

පිළිතුර:
- ගව පාලකයාට තේරුම් ගත හැකි ලෙස
- අවශ්‍ය නම් උදාහරණ සහ හේතු සමඟ
- ප්‍රායෝගික උපදෙස් ලෙස
- සම්පූර්ණ හා පැහැදිලි පිළිතුරක්
- පිළිතුර අවසන් වූ විට නතර වන්න
"""

# ------------------------------------------------------------------
# RAG Context Retrieval
# ------------------------------------------------------------------
def get_frozen_context_debug(query: str, top_k: int = 10, max_chars: int = 6000):
    # Skip intent filtering for debugging
    filtered_docs = docs

    embeddings = []
    for doc in filtered_docs:
        ids = encode_text(doc.page_content, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(ids).cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings.append(emb[0])

    embeddings = np.array(embeddings)

    q_ids = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model(q_ids).cpu().numpy()
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    _, indices = index.search(q_emb, top_k)
    chunks = []
    total_len = 0
    for i in indices[0]:
        text = filtered_docs[i].page_content.strip()
        if text not in chunks and total_len + len(text) <= max_chars:
            chunks.append(text)
            total_len += len(text)

    print("Retrieved chunks for debugging:")
    for c in chunks:
        print("-", c[:200])  # print first 200 chars
    return "\n\n".join(chunks)


# ------------------------------------------------------------------
# RAG Answer using Gemini
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# RAG Answer using Gemini (Corrected for New SDK)
# ------------------------------------------------------------------
def rag_answer(query: str, top_k: int = 10) -> str:
    context = get_frozen_context_debug(query, top_k=top_k)
    prompt = build_prompt(context, query)

    try:
        # ✅ CORRECT METHOD: client.models.generate_content
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Use "gemini-1.5-flash" or "gemini-2.0-flash-exp" if needed
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8000
            )
        )

        # ✅ CORRECT PROPERTY: .text (not .output_text)
        return response.text.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    queries = [
        "ශ්‍රී ලංකාවේ ගවයන්ට ලබාදෙන ප්‍රධාන ආහාර කුමක්ද?",
        "කිරි වැඩි කිරීම සඳහා ගව ආහාර සැලසුමක්",
        "ගවයන්ට ජලය නොලැබුනහොත් සිදුවන ප්‍රතිඵල"
    ]

    for q in queries:
        print("\nQuery:", q)
        print("Answer:", rag_answer(q))
