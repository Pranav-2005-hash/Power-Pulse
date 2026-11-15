from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.storage.state_manager import state_manager

router = APIRouter(tags=["Model Training"])


class TrainModelRequest(BaseModel):
    epochs: Optional[int] = 12
    batch_size: Optional[int] = 64


@router.post("/train/lstm")
def train_lstm_model(req: Optional[TrainModelRequest] = None):
    """
    Trains LSTM sequential deep learning model:
    Input(24, 1) -> LSTM(32) -> Dense(16, ReLU) -> Dense(1)
    Evaluates in kW units and calculates 95th percentile validation positive residual threshold.
    """
    epochs = req.epochs if req and req.epochs else 12
    batch_size = req.batch_size if req and req.batch_size else 64

    # Enforce safe bounds for academic demo
    epochs = min(max(epochs, 1), 30)

    try:
        results = state_manager.train_single_architecture(
            model_type="lstm",
            epochs=epochs,
            batch_size=batch_size
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training error: {str(e)}")


@router.post("/train/gru")
def train_gru_model(req: Optional[TrainModelRequest] = None):
    """
    Trains GRU sequential deep learning model:
    Input(24, 1) -> GRU(32) -> Dense(16, ReLU) -> Dense(1)
    Evaluates in kW units and calculates 95th percentile validation positive residual threshold.
    """
    epochs = req.epochs if req and req.epochs else 12
    batch_size = req.batch_size if req and req.batch_size else 64

    epochs = min(max(epochs, 1), 30)

    try:
        results = state_manager.train_single_architecture(
            model_type="gru",
            epochs=epochs,
            batch_size=batch_size
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training error: {str(e)}")


@router.post("/train/compare")
def train_and_compare_both(req: Optional[TrainModelRequest] = None):
    """
    Trains both LSTM and GRU models sequentially on identical splits.
    Selects best architecture strictly based on Validation MAE.
    Returns comparison matrix, predictions, and anomalies.
    """
    epochs = req.epochs if req and req.epochs else 12
    batch_size = req.batch_size if req and req.batch_size else 64

    epochs = min(max(epochs, 1), 30)

    try:
        results = state_manager.train_and_compare_both(
            epochs=epochs,
            batch_size=batch_size
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison training error: {str(e)}")

