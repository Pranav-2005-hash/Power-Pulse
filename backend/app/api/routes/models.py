from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any

from app.storage.state_manager import state_manager
from app.schemas.models import (
    TrainRequest, TrainStatusResponse, ModelComparisonResponse, ModelMetrics
)
from app.services.training_service import run_training_background_job

router = APIRouter(prefix="/models", tags=["Models"])

@router.post("/train", response_model=TrainStatusResponse)
def trigger_training(req: TrainRequest):
    if not state_manager.is_dataset_loaded:
        raise HTTPException(status_code=400, detail="Cannot train models without a loaded dataset.")

    if state_manager.train_status["status"] == "training":
        raise HTTPException(status_code=409, detail="A training job is already in progress.")

    try:
        run_training_background_job(
            epochs=req.epochs,
            batch_size=req.batch_size,
            patience=req.patience
        )
        return TrainStatusResponse(**state_manager.train_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/status", response_model=TrainStatusResponse)
def get_training_status():
    return TrainStatusResponse(**state_manager.train_status)

@router.get("/metrics")
def get_model_metrics():
    if not state_manager.models_trained:
        raise HTTPException(status_code=400, detail="No models trained yet. Train models first.")

    # Return clean metrics without large raw arrays
    clean_metrics = {}
    for name, m in state_manager.metrics.items():
        clean_metrics[name] = {
            k: v for k, v in m.items() 
            if k not in ["val_residuals", "test_predictions", "test_actuals"]
        }
    return clean_metrics

@router.get("/compare", response_model=ModelComparisonResponse)
def get_model_comparison():
    if not state_manager.models_trained:
        return ModelComparisonResponse(
            is_trained=False,
            best_model=None,
            selection_reason="Models have not yet been trained. Please trigger training from the Models page."
        )

    model_metrics_map = {}
    for name, m in state_manager.metrics.items():
        model_metrics_map[name] = ModelMetrics(
            model_name=m["model_name"],
            test_mae=m["test_mae"],
            test_rmse=m["test_rmse"],
            test_r2=m["test_r2"],
            val_mae=m["val_mae"],
            val_rmse=m["val_rmse"],
            training_duration_seconds=m["training_duration_seconds"],
            inference_latency_ms=m["inference_latency_ms"],
            epochs_trained=m["epochs_trained"],
            loss_history=m.get("loss_history", []),
            architecture=m.get("architecture", {})
        )

    return ModelComparisonResponse(
        is_trained=True,
        best_model=state_manager.best_model_name,
        selection_reason=state_manager.selection_reason,
        models=model_metrics_map,
        recommendation=(
            f"The deep learning pipeline selected {state_manager.best_model_name} as the production model "
            f"due to superior generalization performance on held-out validation sequences."
        )
    )
