from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from app.services.report_service import (
    generate_forecast_csv, generate_anomalies_csv, generate_model_comparison_csv,
    get_processed_data_csv, generate_evaluation_json, generate_html_summary_report
)

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/forecast.csv")
def download_forecast_csv():
    try:
        csv_data = generate_forecast_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=powerpulse_forecast.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/anomalies.csv")
def download_anomalies_csv():
    try:
        csv_data = generate_anomalies_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=powerpulse_anomalies.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model-comparison.csv")
def download_model_comparison_csv():
    try:
        csv_data = generate_model_comparison_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=powerpulse_model_comparison.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/processed-data.csv")
def download_processed_data_csv():
    try:
        csv_data = get_processed_data_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=powerpulse_hourly_consumption.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/evaluation.json")
def download_evaluation_json():
    try:
        eval_data = generate_evaluation_json()
        return JSONResponse(
            content=eval_data,
            headers={"Content-Disposition": "attachment; filename=powerpulse_evaluation.json"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/summary.html", response_class=HTMLResponse)
def view_html_summary():
    try:
        return HTMLResponse(content=generate_html_summary_report())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
