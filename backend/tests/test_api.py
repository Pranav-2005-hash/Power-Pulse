import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.storage.state_manager import state_manager
from app.core.config import SAMPLE_DATA_PATH

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

    # Also test /api/health
    res_api = client.get("/api/health")
    assert res_api.status_code == 200

def test_upload_sample_and_preview(client):
    # Test loading sample dataset
    res_sample = client.post("/upload/sample")
    assert res_sample.status_code == 200
    data = res_sample.json()
    assert data["is_valid"] is True
    assert data["total_rows"] > 0
    assert "Global_active_power" in data["columns"]
    assert len(data["preview"]) == 10

    # Test GET /dataset/preview
    res_preview = client.get("/dataset/preview")
    assert res_preview.status_code == 200
    pdata = res_preview.json()
    assert pdata["is_loaded"] is True
    assert pdata["total_rows"] > 0
    assert len(pdata["preview"]) == 10

def test_preprocess_endpoint(client):
    # Ensure sample is loaded first
    client.post("/upload/sample")

    res_prep = client.post("/preprocess", json={"target_column": "Global_active_power"})
    assert res_prep.status_code == 200
    pdata = res_prep.json()
    assert pdata["status"] == "success"
    assert pdata["hourly_records"] > 0
    assert pdata["train_samples"] > 0
    assert pdata["val_samples"] > 0
    assert pdata["test_samples"] > 0
    assert pdata["scaler_train_min"] is not None

def test_train_lstm_endpoint(client):
    # Fast 1-epoch test
    res = client.post("/train/lstm", json={"epochs": 1, "batch_size": 64})
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "LSTM"
    assert "metrics" in data
    assert data["metrics"]["mae"] > 0
    assert data["metrics"]["rmse"] > 0
    assert data["anomaly_threshold_kw"] > 0
    assert len(data["sample_predictions"]) > 0

def test_results_endpoints_and_downloads(client):
    # GET /results/predictions
    res_pred = client.get("/results/predictions?limit=50")
    assert res_pred.status_code == 200
    pdata = res_pred.json()
    assert len(pdata["predictions"]) <= 50
    assert "actual" in pdata["predictions"][0]
    assert "predicted" in pdata["predictions"][0]

    # GET /results/anomalies
    res_anom = client.get("/results/anomalies")
    assert res_anom.status_code == 200
    adata = res_anom.json()
    assert "total_anomalies" in adata
    assert "disclaimer" in adata

    # Downloads
    res_dl_pred = client.get("/download/predictions")
    assert res_dl_pred.status_code == 200
    assert "actual" in res_dl_pred.text.lower()

    res_dl_anom = client.get("/download/anomalies")
    assert res_dl_anom.status_code == 200

def test_train_compare_endpoint(client):
    # Fast 1-epoch compare test
    res = client.post("/train/compare", json={"epochs": 1, "batch_size": 64})
    assert res.status_code == 200
    data = res.json()
    assert len(data["comparison_table"]) == 2
    models = [row["model"] for row in data["comparison_table"]]
    assert "LSTM" in models
    assert "GRU" in models
    assert data["best_model"] in ["LSTM", "GRU"]
    assert "best_model_reason" in data

    # Test download comparison CSV
    res_dl_comp = client.get("/download/comparison")
    assert res_dl_comp.status_code == 200
    assert "LSTM" in res_dl_comp.text
    assert "GRU" in res_dl_comp.text
