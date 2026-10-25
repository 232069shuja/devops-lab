from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
TASK = BUNDLE.get("task", "classification")
CLASSES = BUNDLE.get("classes", ["Non-Toxic", "Toxic"])

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        features=len(FEATURES),
        task=TASK,
        feature_names=FEATURES,
        classes=CLASSES,
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400

    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return (
            jsonify(
                error="Input schema mismatch", missing=missing, extra=extra
            ),
            400,
        )

    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        raw_pred = MODEL.predict(frame)[0]
        prediction = (
            raw_pred.item() if hasattr(raw_pred, "item") else int(raw_pred)
        )
        label = (
            CLASSES[prediction]
            if isinstance(prediction, int) and prediction < len(CLASSES)
            else str(prediction)
        )

        result = {
            "prediction": prediction,
            "label": label,
            "task": TASK,
            "disclaimer": (
                "Research/teaching prototype only. Based on Jigsaw Unintended"
                " Bias Toxicity modeling (Khan 2023)."
            ),
        }
        if TASK == "classification" and hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(frame)[0].tolist()
            result["probabilities"] = probs
            result["class_probabilities"] = {
                cls_name: round(prob, 4)
                for cls_name, prob in zip(CLASSES, probs)
            }
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
