from pathlib import Path


def test_source_files_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "backend/app.py").exists(), "backend/app.py must exist"
    assert (
        root / "backend/models/model_bundle.joblib"
    ).exists(), "model_bundle.joblib must exist"
    assert (
        root / "data/sample_inputs.csv"
    ).exists(), "sample_inputs.csv must exist"
    assert (
        root / "backend/requirements.txt"
    ).exists(), "requirements.txt must exist"
