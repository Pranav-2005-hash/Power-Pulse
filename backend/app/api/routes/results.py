from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse

from app.storage.state_manager import state_manager

router = APIRouter(tags=["Results & Downloads"])


@router.get("/results/predictions")
def get_predictions(limit: Optional[int] = 500):
    """
    Returns actual and predicted electricity consumption readings on held-out test data.
    """
    if not state_manager.is_trained:
        raise HTTPException(
            status_code=400,
            detail="No model has been trained yet. Please train LSTM or GRU first."
        )

    preds = state_manager.get_predictions(limit=limit)
    return {
        "model_name": state_manager.current_model_name,
        "total_test_samples": len(state_manager.predictions),
        "returned_count": len(preds),
        "predictions": preds
    }


@router.get("/results/anomalies")
def get_anomalies():
    """
    Returns detected consumption anomalies on test set based on the
    95th percentile of positive validation residuals.
    """
    if not state_manager.is_trained:
        raise HTTPException(
            status_code=400,
            detail="No model has been trained yet. Please train LSTM or GRU first."
        )

    return state_manager.get_anomalies()


@router.get("/results/comparison")
def get_model_comparison():
    """
    Returns side-by-side performance comparison for LSTM vs GRU.
    """
    if not state_manager.comparison_table:
        raise HTTPException(
            status_code=400,
            detail="Models have not been compared yet. Please click 'Compare Both'."
        )

    return state_manager.get_comparison()


@router.get("/results/forecast/24h")
@router.post("/results/forecast/24h")
def get_24h_forecast(model: Optional[str] = None):
    """
    Returns roll-forward 24-hour future electricity consumption predictions
    using the best model (or specified model) starting from the end of the dataset.
    """
    if not state_manager.is_trained:
        raise HTTPException(
            status_code=400,
            detail="No model has been trained yet. Please train LSTM or GRU first."
        )

    try:
        forecast = state_manager.predict_next_24_hours(model_choice=model)
        return forecast
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"24-hour forecast error: {str(e)}")


# CSV Downloads
@router.get("/download/predictions")
def download_predictions_csv():
    """
    Streams CSV of actual vs predicted consumption on test set.
    """
    csv_content = state_manager.generate_predictions_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=powerpulse_predictions.csv"}
    )


@router.get("/download/anomalies")
def download_anomalies_csv():
    """
    Streams CSV of detected electricity consumption anomalies.
    """
    csv_content = state_manager.generate_anomalies_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=powerpulse_anomalies.csv"}
    )


@router.get("/download/comparison")
def download_comparison_csv():
    """
    Streams CSV of LSTM vs GRU comparison metrics.
    """
    csv_content = state_manager.generate_comparison_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=powerpulse_model_comparison.csv"}
    )


@router.get("/download/forecast")
def download_forecast_csv():
    """
    Streams CSV of the 24-hour future electricity consumption forecast.
    """
    csv_content = state_manager.generate_forecast_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=powerpulse_24h_forecast.csv"}
    )

