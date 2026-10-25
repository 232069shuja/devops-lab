import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import FEATURES, app

PAYLOAD = json.loads(
    (ROOT / "backend/test_payload.json").read_text(encoding="utf-8")
)


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["features"] == len(FEATURES)


def test_predict_valid():
    response = app.test_client().post("/predict", json=PAYLOAD)
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert "prediction" in data
    assert "label" in data


def test_predict_missing_feature():
    incomplete = dict(PAYLOAD)
    incomplete.pop(FEATURES[0])
    response = app.test_client().post("/predict", json=incomplete)
    assert response.status_code == 400
    data = response.get_json()
    assert "missing" in data


def test_predict_invalid_json():
    response = app.test_client().post(
        "/predict", data="not-json", content_type="application/json"
    )
    assert response.status_code == 400
