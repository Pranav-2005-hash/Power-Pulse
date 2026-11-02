import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import SAMPLE_DATA_PATH
from app.services.sample_generator import generate_sample_dataset
from app.storage.state_manager import state_manager
from app.api.routes import health, dataset, analytics, models, forecast, anomalies, reports

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure sample dataset exists and initialize
    print("PowerPulse Backend initializing...")
    if not os.path.exists(SAMPLE_DATA_PATH):
        print("Generating realistic UCI household power sample dataset...")
        generate_sample_dataset(str(SAMPLE_DATA_PATH), days=90)

    if not state_manager.is_dataset_loaded:
        print("Preloading default sample dataset into StateManager...")
        try:
            state_manager.load_dataset_from_file(str(SAMPLE_DATA_PATH), source_type="sample")
            print("Default sample dataset loaded successfully.")
        except Exception as e:
            print(f"Warning during sample preloading: {e}")

    yield
    print("PowerPulse Backend shutting down...")

app = FastAPI(
    title="PowerPulse: Deep Learning Energy Analytics API",
    description="REST API for deep learning energy forecasting and residual-based anomaly detection",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development and frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes with /api prefix
app.include_router(health.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(anomalies.router, prefix="/api")
app.include_router(reports.router, prefix="/api")

@app.get("/")
def root():
    return {
        "app": "PowerPulse Deep Learning Engine",
        "docs": "/docs",
        "health": "/api/health"
    }
