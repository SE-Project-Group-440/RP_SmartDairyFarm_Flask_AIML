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
import itertools


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

# Collect all GROQ_API_KEY_N keys
api_keys = []
i = 1
while True:
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if not key:
        break
    api_keys.append(key)
    i += 1

if not api_keys:
    raise ValueError("No GROQ_API_KEY_N variables found in .env")

def groq_client_factory():
    for key in api_keys:
        yield Groq(api_key=key)

clients = groq_client_factory()
clients_cycle = itertools.cycle(api_keys)

def groq_chat_request(messages, model="llama-3.3-70b-versatile", temperature=0.35, max_tokens=2048):
    
    while True:  
        key = next(clients_cycle)  
        client = Groq(api_key=key)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response

        except Exception as e:
            err_msg = str(e).lower()

            if "quota" in err_msg or "tokens" in err_msg or "rate limit" in err_msg:
                print(f"API key exhausted or rate limited: {key}. Switching key...")
                continue 

            else:
                print(f"Unexpected error with key {key}: {e}")
                continue

def build_prompt(context: str, query: str) -> str:
    return f"""
ඔබ ශ්‍රී ලංකාවේ කිරි ගව පාලකයින්ට උපදෙස් ලබාදෙන කෘෂිකාර්මික උපදේශකයෙකි.

පහත දැනුම් පදනම භාවිතා කරමින් ප්‍රශ්නයට පිළිතුරු ලබාදෙන්න.

දැනුම් පදනම:
{context}

ප්‍රශ්නය:
{query}

පිළිතුර ලබාදීමේ නීති:

1. එකම අදහස නැවත නැවත නොලියන්න
2. සරල සිංහල භාෂාව භාවිතා කරන්න
3. ගව පාලකයාට ප්‍රායෝගික උපදෙස් ලබාදෙන්න
4. පිළිතුර අවසානයේ සම්පූර්ණ වාක්‍යයකින් අවසන් කරන්න
5. පිළිතුර විස්තරාත්මක විය යුතුය, අවම වශයෙන් 5–6 කොටස්වලට බෙදා ලියන්න
6. Markdown (** , ## , - ) භාවිතා නොකරන්න
7. අනිවාර්යයෙන් පිළිතුර අවසන් වන්නේ සම්පූර්ණ වාක්‍යයකින් විය යුතුය. වාක්‍ය අඩකින් නවත්වන්නම එපා.

"""

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

def remove_repeated_phrases(text):
    words = text.split()
    cleaned = []
    
    for w in words:
        if len(cleaned) > 5 and w == cleaned[-1]:
            continue
        cleaned.append(w)

    return " ".join(cleaned)

def rag_answer(query: str, top_k: int = 10):
    # Retrieve
    model.eval()
    q_ids = encode_text(query, bpe_merges, bpe_vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model(q_ids).cpu().numpy()
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    distances, indices = index.search(q_emb, top_k)
    retrieved_docs = [docs[i] for i in indices[0]]

    unique_chunks = list(
        dict.fromkeys(
            [doc.page_content.strip()[:800] for doc in retrieved_docs]
        )
    )
    
    if not retrieved_docs:
        return "මෙම ප්‍රශ්නයට සම්බන්ධ තොරතුරු දැනුම් පදනමේ නොමැත."
    
    context = "\n\n".join(unique_chunks)
    context = deduplicate_sentences(context)

    prompt = build_prompt(context, query)
    response = groq_chat_request(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
        max_tokens=2048
    )
    answer = response.choices[0].message.content.strip()
    answer = deduplicate_sentences(answer)
    answer = remove_repeated_phrases(answer)
    return answer
