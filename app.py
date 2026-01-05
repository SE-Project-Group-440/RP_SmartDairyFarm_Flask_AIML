from fastapi import FastAPI
from routes.prediction_routes import router as prediction_router
from routes.heat_prediction_routes import router as heat_router

app = FastAPI()

app.include_router(prediction_router, prefix="/api")
app.include_router(heat_router, prefix="/api")
