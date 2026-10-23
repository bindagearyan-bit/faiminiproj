"""
app.py

Interactive Clinical Heart Disease Classification Web Application.
Serves a responsive single-page diagnostic dashboard allowing faculty and students
to input ANY arbitrary clinical values, run live model inference with zero latency,
and dynamically render exactly ONE tailored visualization (Matplotlib / Seaborn / Scikit-Learn)
that most effectively depicts the patient's diagnostic outcome.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    FEATURE_METADATA,
    FEATURE_NAMES,
    PRESET_PROFILES,
    load_model_pipeline,
    predict_patient,
)
from src.visualization import select_best_visualization

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

# Pre-load ML pipeline in memory at startup
print("[STARTUP] Loading trained Scikit-learn Heart Disease pipeline...")
PIPELINE = load_model_pipeline()
print("[STARTUP] Pipeline successfully loaded and ready for live inference.")


AVAILABLE_VISUALIZATIONS = [
    {
        "id": "auto",
        "title": "✨ Auto (Model AI Choice)",
        "description": "Selects the single best visualization depicting this patient's findings",
    },
    {
        "id": "biomarker_heatmap",
        "title": "🔥 Biomarker Risk Heatmap",
        "description": "Seaborn multi-marker normalized thermal risk intensity matrix",
    },
    {
        "id": "feature_impact",
        "title": "📊 ML Feature Attribution (Log-Odds)",
        "description": "Scikit-Learn pipeline linear weights & coefficient attribution",
    },
    {
        "id": "ecg_ischemia",
        "title": "📈 ECG ST-Segment Waveform",
        "description": "Matplotlib exercise ST-depression & depolarization ischemia curve",
    },
    {
        "id": "age_thalach_kde",
        "title": "🎯 Cardiac Stress Space (Age vs HR)",
        "description": "Seaborn bivariate KDE distribution & age-predicted max HR zone",
    },
]


@app.route("/")
def index():
    """Renders the primary interactive diagnostic dashboard."""
    return render_template(
        "index.html",
        feature_metadata=FEATURE_METADATA,
        feature_names=FEATURE_NAMES,
        presets=PRESET_PROFILES,
        available_viz=AVAILABLE_VISUALIZATIONS,
    )


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Returns preset clinical profiles for demonstration."""
    return jsonify(
        {
            "status": "success",
            "presets": PRESET_PROFILES,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts arbitrary custom patient data and performs live ML inference.
    Dynamically generates exactly 1 visualization from Matplotlib/Seaborn/Scikit-learn
    that best depicts this patient's specific results.
    """
    start_time = time.perf_counter()
    data: Dict[str, Any] = request.get_json(force=True) or {}

    override_viz = data.get("override_viz", "auto")
    override_id = None if override_viz in ["auto", "", None] else override_viz

    # 1. Perform ML prediction on custom values
    pred_result = predict_patient(data, pipeline=PIPELINE)

    # 2. Generate the single best visualization
    viz_result = select_best_visualization(
        pipeline=PIPELINE,
        patient_dict=pred_result["input_features"],
        prediction=pred_result["prediction"],
        prob_disease=pred_result["prob_disease"],
        override_id=override_id,
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return jsonify(
        {
            "status": "success",
            "inference_time_ms": elapsed_ms,
            "prediction": pred_result["prediction"],
            "label": pred_result["label"],
            "prob_disease": round(pred_result["prob_disease"] * 100, 1),
            "prob_no_disease": round(pred_result["prob_no_disease"] * 100, 1),
            "risk_tier": pred_result["risk_tier"],
            "key_factors": pred_result["key_factors"],
            "input_features": pred_result["input_features"],
            "visualization": viz_result,
            "available_visualizations": AVAILABLE_VISUALIZATIONS,
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": PIPELINE is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n" + "=" * 70)
    print(f"[SERVER ONLINE] HEART DISEASE DIAGNOSTIC WEB APPLICATION RUNNING")
    print(f"[URL] Local Web URL: http://127.0.0.1:{port}")
    print(f"[URL] Alternative:   http://localhost:{port}")
    print(f"=" * 70 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
