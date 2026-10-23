"""
visualization.py

Generates authentic, high-impact clinical data visualizations using
matplotlib, seaborn, and scikit-learn.
Engineered for academic viva examinations & real clinical dashboard demonstrations:
crystal-clear dark-mode aesthetic, authentic statistical distributions,
and zero artificial or confusing clutter.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_cleveland.csv"

_DATASET_CACHE: Optional[pd.DataFrame] = None


def get_dataset() -> Optional[pd.DataFrame]:
    global _DATASET_CACHE
    if _DATASET_CACHE is None and DATA_PATH.exists():
        _DATASET_CACHE = pd.read_csv(DATA_PATH)
    return _DATASET_CACHE


def fig_to_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib figure into base64 string with crisp dark-mode rendering."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#0e172a", edgecolor="#1e293b")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def set_colab_theme(ax: plt.Axes, fig: plt.Figure):
    """Configures clean, authentic JupyterLab / Colab Dark Mode styling."""
    fig.patch.set_facecolor("#0e172a")
    ax.set_facecolor("#111c35")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)
    ax.xaxis.label.set_color("#f1f5f9")
    ax.yaxis.label.set_color("#f1f5f9")
    ax.title.set_color("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.35, color="#1e293b")


def generate_biomarker_heatmap(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders a standard Seaborn heatmap comparing the patient's key numbers
    with normal healthy reference values.
    """
    metrics = [
        ("Resting BP", float(patient_dict.get("trestbps", 120)), 120.0, "mm Hg"),
        ("Cholesterol", float(patient_dict.get("chol", 200)), 200.0, "mg/dL"),
        ("Max Heart Rate", float(patient_dict.get("thalach", 150)), 155.0, "bpm"),
        ("ECG Stress Reading", float(patient_dict.get("oldpeak", 0)), 0.0, "score"),
        ("Blocked Vessels", float(patient_dict.get("ca", 0)), 0.0, "count"),
    ]

    col_names = [f"{m[0]}\n({m[3]})" for m in metrics]
    patient_vals = [m[1] for m in metrics]
    normal_vals = [m[2] for m in metrics]

    # Calculate risk ratio: > 1.0 means elevated risk
    risk_ratios = []
    for i, m in enumerate(metrics):
        p_val = m[1]
        norm = m[2]
        if m[0] == "Max Heart Rate":
            # For heart rate, lower than normal is risky
            ratio = (norm - p_val) / 40.0 + 0.5
        elif norm == 0.0:
            ratio = 1.0 + (p_val * 0.5) if p_val > 0 else 0.2
        else:
            ratio = p_val / norm
        risk_ratios.append(min(max(ratio, 0.2), 2.0))

    matrix = np.array([patient_vals, normal_vals, risk_ratios])

    annot = np.array([
        [f"{v:.1f}" for v in patient_vals],
        [f"{v:.1f}" for v in normal_vals],
        [f"{'HIGH' if r > 1.15 else ('NORMAL' if r < 0.95 else 'BORDERLINE')}" for r in risk_ratios]
    ])

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.patch.set_facecolor("#0e172a")
    ax.set_facecolor("#0e172a")

    sns.heatmap(
        matrix[:2],
        annot=annot[:2],
        fmt="",
        cmap="rocket_r",
        linewidths=2.0,
        linecolor="#0e172a",
        cbar=False,
        xticklabels=col_names,
        yticklabels=["Your Patient Value", "Healthy Benchmark"],
        annot_kws={"size": 11, "weight": "bold", "color": "#f8fafc"},
        ax=ax,
    )

    ax.set_title("Seaborn Heatmap: Patient Health Values vs. Healthy Baseline", fontsize=11.5, fontweight="bold", pad=12, color="#f8fafc")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)

    base64_str = fig_to_base64(fig)
    return {
        "id": "biomarker_heatmap",
        "title": "🔥 Biomarker Heatmap (Patient vs Healthy Baseline)",
        "library_used": "Seaborn (sns.heatmap) & Matplotlib",
        "reason": "Compares the patient's blood pressure, cholesterol, and heart rate directly with standard healthy benchmarks on a Seaborn thermal scale.",
        "image_base64": base64_str,
    }


def generate_feature_impact_plot(pipeline, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders a standard horizontal bar chart showing which inputs
    pushed the model toward Heart Disease (Red) vs Healthy (Green).
    """
    from src.predict import FEATURE_NAMES

    row_dict = {col: patient_dict.get(col, 0) for col in FEATURE_NAMES}
    df_patient = pd.DataFrame([row_dict], columns=FEATURE_NAMES)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(df_patient)[0]
    coefs = classifier.coef_[0]
    feature_names = preprocessor.get_feature_names_out()

    contributions = X_trans * coefs

    friendly_names = {
        "num__age": f"Age ({patient_dict.get('age', 0):.0f} yrs)",
        "num__trestbps": f"Blood Pressure ({patient_dict.get('trestbps', 0):.0f} mmHg)",
        "num__chol": f"Cholesterol ({patient_dict.get('chol', 0):.0f} mg/dL)",
        "num__thalach": f"Max Heart Rate ({patient_dict.get('thalach', 0):.0f} bpm)",
        "num__oldpeak": f"ECG Stress Reading ({patient_dict.get('oldpeak', 0):.1f})",
        "cat__sex_1.0": "Male Gender",
        "cat__sex_0.0": "Female Gender",
        "cat__cp_1.0": "Severe Chest Pain (Type 1)",
        "cat__cp_2.0": "Moderate Chest Pain (Type 2)",
        "cat__cp_3.0": "Mild Chest Pain (Type 3)",
        "cat__cp_4.0": "No Chest Pain / Silent (Type 4)",
        "cat__fbs_1.0": "Blood Sugar > 120 (High)",
        "cat__fbs_0.0": "Blood Sugar <= 120 (Normal)",
        "cat__restecg_0.0": "Normal Resting ECG",
        "cat__restecg_1.0": "ECG Wave Issue",
        "cat__restecg_2.0": "Heart Muscle Enlarged",
        "cat__exang_1.0": "Exercise Chest Pain: Yes",
        "cat__exang_0.0": "Exercise Chest Pain: No",
        "cat__slope_1.0": "ECG Slope: Upsloping",
        "cat__slope_2.0": "ECG Slope: Flat",
        "cat__slope_3.0": "ECG Slope: Downsloping",
        "cat__ca_0.0": "Zero Blocked Vessels",
        "cat__ca_1.0": "1 Blocked Vessel",
        "cat__ca_2.0": "2 Blocked Vessels",
        "cat__ca_3.0": "3 Blocked Vessels",
        "cat__thal_3.0": "Normal Blood Flow",
        "cat__thal_6.0": "Partial Blood Block",
        "cat__thal_7.0": "Severe Blood Block",
    }

    records = []
    for name, impact, val in zip(feature_names, contributions, X_trans):
        if abs(impact) > 0.001 or abs(val) > 0.001:
            clean_title = friendly_names.get(name, name)
            records.append((clean_title, impact))

    records.sort(key=lambda x: abs(x[1]), reverse=True)
    top_records = records[:7]
    top_records.reverse()

    names = [r[0] for r in top_records]
    vals = [r[1] for r in top_records]
    colors = ["#f43f5e" if v > 0 else "#10b981" for v in vals]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    set_colab_theme(ax, fig)

    bars = ax.barh(names, vals, color=colors, height=0.55, edgecolor="#1e293b", linewidth=0.7)
    ax.axvline(0, color="#f8fafc", linestyle="-", linewidth=1.1)

    for bar, val in zip(bars, vals):
        offset = 0.04 if val >= 0 else -0.04
        ha = "left" if val >= 0 else "right"
        sign = "+" if val > 0 else ""
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.2f}",
            va="center",
            ha=ha,
            color="#f8fafc",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title("Scikit-Learn Model: What Pushed the Prediction? (Feature Weights)", fontsize=11.5, fontweight="bold", pad=12, color="#f8fafc")
    ax.set_xlabel("Impact on Model Prediction (Red = Disease Risk | Green = Healthy Indicator)", fontsize=9.5, labelpad=6, color="#cbd5e1")

    max_abs = max([abs(v) for v in vals] + [0.8])
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    base64_str = fig_to_base64(fig)
    return {
        "id": "feature_impact",
        "title": "📊 Model Feature Weights (What Decided This Result?)",
        "library_used": "Scikit-Learn (Model Weights) & Matplotlib",
        "reason": "Shows which patient inputs pushed the model towards Heart Disease (Red) vs Healthy (Green) based on Scikit-Learn weights.",
        "image_base64": base64_str,
    }


def generate_age_thalach_kde_plot(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders a standard Seaborn scatter plot comparing this patient
    against the actual dataset records (Healthy vs Heart Disease).
    """
    age = float(patient_dict.get("age", 55.0))
    thalach = float(patient_dict.get("thalach", 150.0))

    df = get_dataset()

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    set_colab_theme(ax, fig)

    if df is not None and "age" in df.columns and "thalach" in df.columns and "target" in df.columns:
        sns.scatterplot(
            data=df,
            x="age",
            y="thalach",
            hue="target",
            palette={0: "#38bdf8", 1: "#f43f5e"},
            alpha=0.65,
            s=48,
            edgecolor="#0e172a",
            linewidth=0.5,
            ax=ax,
        )

    # Patient position
    ax.scatter(
        [age],
        [thalach],
        color="#fbbf24",
        s=220,
        marker="*",
        edgecolor="#ffffff",
        linewidth=1.2,
        zorder=10,
        label=f"Your Patient (Age {age:.0f}, Heart Rate {thalach:.0f})",
    )

    ax.set_title("Seaborn Scatter: Where Does Your Patient Fall in the Dataset?", fontsize=11.5, fontweight="bold", pad=12, color="#f8fafc")
    ax.set_xlabel("Patient Age (Years)", fontsize=9.5, color="#cbd5e1")
    ax.set_ylabel("Maximum Heart Rate Achieved (bpm)", fontsize=9.5, color="#cbd5e1")

    handles, labels = ax.get_legend_handles_labels()
    custom_labels = []
    for lbl in labels:
        if lbl == "0":
            custom_labels.append("Dataset: Healthy Patients (0)")
        elif lbl == "1":
            custom_labels.append("Dataset: Heart Disease Patients (1)")
        else:
            custom_labels.append(lbl)
    ax.legend(handles, custom_labels, loc="lower left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    base64_str = fig_to_base64(fig)
    return {
        "id": "age_thalach_kde",
        "title": "🎯 Patient vs Dataset (Age vs Max Heart Rate)",
        "library_used": "Seaborn (sns.scatterplot) & Matplotlib",
        "reason": "Plots this patient as a star against all 303 real patients from the Cleveland dataset to see where they fall.",
        "image_base64": base64_str,
    }


def generate_probability_meter_plot(patient_dict: Dict[str, Any], prob_disease: float) -> Dict[str, Any]:
    """
    Renders a standard Matplotlib horizontal decision confidence gauge
    with the 50% classification cutoff threshold.
    """
    prob_pct = prob_disease * 100.0
    is_cad = prob_disease >= 0.5

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    set_colab_theme(ax, fig)

    # Horizontal bar from 0 to 100%
    bar_color = "#f43f5e" if is_cad else "#10b981"
    ax.barh(["Prediction\nConfidence"], [prob_pct], color=bar_color, height=0.45, edgecolor="#0e172a", linewidth=0.8)

    # 50% threshold line
    ax.axvline(50, color="#f8fafc", linestyle="--", linewidth=1.5, label="Decision Threshold (50%)")

    # Safe vs Risk background zones
    ax.axvspan(0, 50, color="#064e3b", alpha=0.35, label="Healthy Zone (< 50%)")
    ax.axvspan(50, 100, color="#881337", alpha=0.35, label="Heart Disease Zone (>= 50%)")

    ax.text(
        prob_pct / 2,
        0,
        f"{prob_pct:.1f}% Risk",
        va="center",
        ha="center",
        color="#ffffff",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Model Predicted Risk Probability (%)", fontsize=9.5, color="#cbd5e1")
    ax.set_title(f"Classification Output: {'HEART DISEASE (Class 1)' if is_cad else 'HEALTHY (Class 0)'} at {prob_pct:.1f}% Confidence", fontsize=11.5, fontweight="bold", pad=12, color="#f8fafc")
    ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    base64_str = fig_to_base64(fig)
    return {
        "id": "prob_gauge",
        "title": "📈 Model Decision Threshold & Probability Gauge",
        "library_used": "Matplotlib (Decision Boundary Plot)",
        "reason": "Shows the 50% machine learning decision line and how confident the model is in this prediction.",
        "image_base64": base64_str,
    }


def select_best_visualization(
    pipeline,
    patient_dict: Dict[str, Any],
    prediction: int,
    prob_disease: float,
    override_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Intelligently selects ONLY 1 visualization that best depicts the patient's result.
    Simple, intuitive rule for academic presentation:
      - If patient has abnormal BP, Cholesterol, or vessels -> Biomarker Heatmap (Seaborn)
      - If user wants to see what features caused the output -> Feature Weights (Scikit-Learn)
      - If extreme age or heart rate -> Dataset Scatter Plot (Seaborn)
      - Otherwise -> Model Probability Gauge (Matplotlib)
    """
    oldpeak = float(patient_dict.get("oldpeak", 0.0))
    ca = float(patient_dict.get("ca", 0.0))
    chol = float(patient_dict.get("chol", 200.0))
    trestbps = float(patient_dict.get("trestbps", 120.0))
    thalach = float(patient_dict.get("thalach", 150.0))
    age = float(patient_dict.get("age", 55.0))

    if override_id == "biomarker_heatmap" or (
        override_id is None and (ca >= 1.0 or chol >= 240.0 or trestbps >= 140.0)
    ):
        chosen = generate_biomarker_heatmap(patient_dict)
    elif override_id == "feature_impact" or (
        override_id is None and (oldpeak >= 1.5 or patient_dict.get("exang", 0) == 1)
    ):
        chosen = generate_feature_impact_plot(pipeline, patient_dict)
    elif override_id == "age_thalach_kde" or (
        override_id is None and (thalach < 125.0 or age >= 65.0)
    ):
        chosen = generate_age_thalach_kde_plot(patient_dict)
    elif override_id == "prob_gauge" or override_id is None:
        chosen = generate_probability_meter_plot(patient_dict, prob_disease)
    else:
        chosen = generate_feature_impact_plot(pipeline, patient_dict)

    return chosen
