import os
import uuid
from gtts import gTTS
from scripts.rag_query import rag_answer

AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_chat_response(query: str, audioUri: str) -> dict:
    # Get the answer from RAG model
    answer = rag_answer(query)

    # Generate audio
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, filename)

    try:
        tts = gTTS(text=answer, lang="si")
        tts.save(file_path)
        audio_uri = f"{audioUri}/audio_files/{filename}"
    except Exception as e:
        print("Error generating audio:", e)
        audio_uri = ""

    return {"answer": answer, "audioUri": audio_uri}