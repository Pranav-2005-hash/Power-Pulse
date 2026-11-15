import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import SAMPLE_DATA_PATH
from app.services.sample_generator import generate_sample_dataset
from app.api.routes import health, dataset, training, results

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure bundled demo sample dataset exists
    print("PowerPulse Backend initializing...")
    if not os.path.exists(SAMPLE_DATA_PATH):
        print("Generating realistic UCI household power sample dataset...")
        generate_sample_dataset(str(SAMPLE_DATA_PATH), days=90)
    print("PowerPulse Deep Learning Engine ready.")
    yield
    print("PowerPulse Backend shutting down...")

app = FastAPI(
    title="PowerPulse: Electricity Consumption Prediction & Anomaly Detection",
    description="Academic deep learning REST API using LSTM and GRU neural networks",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes directly at root for clean academic endpoints
app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(results.router)

# Also register with /api prefix for compatibility
app.include_router(health.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(results.router, prefix="/api")

@app.get("/")
def root():
    return {
        "project": "PowerPulse",
        "subtitle": "Electricity Consumption Prediction and Anomaly Detection",
        "docs": "/docs",
        "health": "/health"
    }
