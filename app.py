from fastapi import FastAPI
from routes.prediction_routes import router as prediction_router
from routes.ai_model_routes import router as ai_model_router

app = FastAPI()

app.include_router(prediction_router, prefix="/api")
app.include_router(ai_model_router, prefix="/api")
