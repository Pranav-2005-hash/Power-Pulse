import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.storage.state_manager import state_manager
from app.core.config import SAMPLE_DATA_PATH

@pytest.fixture
def client():
    # Ensure sample dataset is loaded into state
    if not state_manager.is_dataset_loaded and SAMPLE_DATA_PATH.exists():
        state_manager.load_dataset_from_file(str(SAMPLE_DATA_PATH), source_type="sample")
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_dataset_status_and_preview(client):
    res_status = client.get("/api/dataset/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["is_loaded"] is True
    assert data["target_column"] == "Global_active_power"

    res_preview = client.get("/api/dataset/preview?page=1&page_size=10")
    assert res_preview.status_code == 200
    pdata = res_preview.json()
    assert len(pdata["rows"]) == 10
    assert "Global_active_power" in pdata["columns"]

def test_analytics_endpoints(client):
    res_sum = client.get("/api/analytics/summary")
    assert res_sum.status_code == 200
    sum_data = res_sum.json()
    assert "stats" in sum_data
    assert sum_data["stats"]["mean"] > 0

    res_series = client.get("/api/analytics/consumption?limit=20")
    assert res_series.status_code == 200
    assert len(res_series.json()) <= 20

    res_patterns = client.get("/api/analytics/patterns")
    assert res_patterns.status_code == 200
    pat_data = res_patterns.json()
    assert len(pat_data["hourly"]) == 24
    assert len(pat_data["daily"]) == 7

def test_forecast_and_reports_error_handling(client):
    res = client.post("/api/forecast", json={"model": "best", "horizon": 24})
    if not state_manager.models_trained:
        assert res.status_code == 400
        assert "not trained" in res.json()["detail"].lower()
    else:
        assert res.status_code == 200
        assert len(res.json()["forecast_points"]) == 24

def test_processed_data_report_download(client):
    res = client.get("/api/reports/processed-data.csv")
    assert res.status_code == 200
    assert "Global_active_power" in res.text
