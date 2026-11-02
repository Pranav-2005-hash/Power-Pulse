from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional

class TrainRequest(BaseModel):
    epochs: Optional[int] = Field(default=15, ge=1, le=50)
    batch_size: Optional[int] = Field(default=64, ge=16, le=256)
    patience: Optional[int] = Field(default=4, ge=1, le=10)

class EpochLog(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float

class ModelMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    test_mae: float
    test_rmse: float
    test_r2: float
    val_mae: float
    val_rmse: float
    training_duration_seconds: float
    inference_latency_ms: float
    epochs_trained: int
    loss_history: List[EpochLog] = []
    architecture: Dict[str, Any] = {}

class TrainStatusResponse(BaseModel):
    status: str  # "idle", "training", "completed", "failed"
    progress_percentage: float
    current_model: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class ModelComparisonResponse(BaseModel):
    is_trained: bool
    best_model: Optional[str] = None
    selection_reason: str
    models: Dict[str, ModelMetrics] = {}
    recommendation: Optional[str] = None
