import os
import torch
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
from scripts.utils import load_all_knowledge, encode_text, MiniEmbeddingModel
from scripts.nlu import load_nlu, classify_intent

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load knowledge base
docs = load_all_knowledge("data/knowledge_base")

# Load BPE tokenizer
with open("models/bpe_merges.pkl", "rb") as f:
    bpe_merges = pickle.load(f)
with open("models/bpe_vocab.pkl", "rb") as f:
    bpe_vocab = pickle.load(f)

# Load embedding model
model = MiniEmbeddingModel(len(bpe_vocab)).to(device)
model.load_state_dict(torch.load("models/sinhala_embedding_model.pt", map_location=device))
model.eval()

# Load Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load NLU model
nlu_model, intent_map = load_nlu(bpe_merges, bpe_vocab)

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
- වාක්‍ය හෝ අදහස් අඩංගු නොවී අඩක් නොනවත්වා
- ප්‍රශ්නයට සම්පූර්ණ සහ විස්තරාත්මක පිළිතුරක් ලබාදෙන්න
- එකම අදහස නැවත නැවත නොලියන්න
- එකම වාක්‍ය රටාව නැවත භාවිත නොකරන්න

"""

def get_frozen_context(query: str, top_k: int = 3, max_chars: int = 6000):

    intent = classify_intent(query, nlu_model, bpe_merges, bpe_vocab, intent_map)

    #Intent-based filtering
    filtered_docs = [
        doc for doc in docs
        if doc.metadata.get("category") == intent
    ]

    if not filtered_docs:
        filtered_docs = docs   

    #Generate embeddings
    embeddings = []
    for doc in filtered_docs:
        text = doc.page_content  
        ids = encode_text(text, bpe_merges, bpe_vocab).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(ids).cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        embeddings.append(emb[0])

    embeddings = np.array(embeddings)

    #Query embedding
    q_ids = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model(q_ids).cpu().numpy()
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    #FAISS search
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

    return "\n\n".join(chunks)


def rag_answer(query: str, top_k: int = 3) -> str:
    """Retrieve context and query LLM."""
    context = get_frozen_context(query, top_k=top_k)
    
    if context.startswith("මෙම ප්‍රශ්නයට"):
        return context  
    
    prompt = build_prompt(context, query)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=8000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"

if __name__ == "__main__":
    queries = [
        "ශ්‍රී ලංකාවේ ගවයන්ට ලබාදෙන ප්‍රධාන ආහාර කුමක්ද?",
        "කිරි වැඩි කිරීම සඳහා ගව ආහාර සැලසුමක්",
        "ගවයන්ට ජලය නොලැබුනහොත් සිදුවන ප්‍රතිඵල"
    ]
    
    for q in queries:
        print("\n Query:", q)
        answer = rag_answer(q)
        print("Answer:", answer)


#token now set for 8000