from fastapi import FastAPI
from routes.prediction_routes import router as prediction_router
from routes.cattle_disease_routes import router as cattle_disease_router

app = FastAPI()

# register routes
app.include_router(prediction_router, prefix="/api")
app.include_router(cattle_disease_router, prefix="/api")


app = FastAPI()

app.include_router(prediction_router, prefix="/api")

