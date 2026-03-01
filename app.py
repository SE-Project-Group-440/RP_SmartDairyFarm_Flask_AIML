from fastapi import FastAPI
from routes.chat_routes import router as chat_router
from routes.stt_routes import router as stt_router
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Sinhala Dairy RAG API")

# Serve audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio_files", StaticFiles(directory=AUDIO_DIR), name="audio_files")

# Include routes
app.include_router(chat_router)
app.include_router(stt_router)