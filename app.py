from fastapi import FastAPI
from routes.chat_routes import router as chat_router
from routes.stt_routes import router as stt_router
from routes.prediction_routes import router as prediction_router
from routes.ai_model_routes import router as ai_model_router
from routes.heat_prediction_routes import router as heat_router
from routes.cattle_disease_routes import router as cattle_disease_router
from routes.pipeline_routes import router as pipeline_router 
from routes.auth_routes import router as auth_router
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Sinhala Dairy RAG API")

AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio_files", StaticFiles(directory=AUDIO_DIR), name="audio_files")

app.include_router(chat_router)
app.include_router(stt_router)
app.include_router(prediction_router, prefix="/api")
app.include_router(ai_model_router, prefix="/api")
app.include_router(heat_router, prefix="/api")
app.include_router(cattle_disease_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api") 
app.include_router(auth_router, prefix="/api")