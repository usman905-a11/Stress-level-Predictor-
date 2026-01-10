import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# =========================
# Paths
# =========================
MODEL_PATH = r"D:\stress_app\stress_lr_multinomial.joblib"
DATA_PATH = r"D:\stress_app\StressLevelDataset_Cleaned.csv"

# =========================
# Human-readable labels
# =========================
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# =========================
# Flask app
# =========================
app = Flask(__name__, template_folder="templates", static_folder="static")

# =========================
# Load model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


df_all = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
features = [c for c in df_all.columns if c != "stress_level"]

def coerce_row_to_numeric(form_dict: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    row = {}
    for f in feature_order:
        val = form_dict.get(f)
        if val is None or val == "":
            raise ValueError(f"Missing value for {f}")
        try:
            row[f] = float(val)
        except Exception:
            raise ValueError(f"Feature {f} must be numeric. Got {val}")
    return pd.DataFrame([row])[feature_order]

def to_prob_dict(classes_arr: np.ndarray, probs_arr: np.ndarray) -> Dict[str, float]:
    return {LABEL_MAP[int(cid)]: round(float(p), 4) for cid, p in zip(classes_arr, probs_arr)}

classes_ = model.classes_

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("interface.html", features=features)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        form_data = {k: request.form.get(k) for k in request.form.keys()}
        x_row = coerce_row_to_numeric(form_data, features)

        # Avoid NaNs breaking the model
        x_row = x_row.fillna(0)

        pred = int(model.predict(x_row)[0])
        probs = model.predict_proba(x_row)[0]

        result = {
            "predicted_class": LABEL_MAP.get(pred, str(pred)),
            "probabilities": to_prob_dict(classes_, probs),
        }

        return render_template(
            "interface.html",
            features=features,
            prediction=pred,
            result=result
        )
    except Exception as e:
        # Always return the same template, with error displayed
        return render_template(
            "interface.html",
            features=features,
            error=f"⚠️ Prediction failed: {str(e)}"
        )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = {k: request.form.get(k) for k in request.form.keys()}
        x_row = coerce_row_to_numeric(form_data, features)
        x_row = x_row.fillna(0)

        pred = int(model.predict(x_row)[0])
        probs = model.predict_proba(x_row)[0]

        result = {
            "predicted_class": LABEL_MAP.get(pred, str(pred)),
            "probabilities": to_prob_dict(classes_, probs),
        }
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)




