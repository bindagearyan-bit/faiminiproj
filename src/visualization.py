"""
visualization.py

Generates dynamic, single-best clinical & ML visualizations on the fly
using matplotlib, seaborn, and scikit-learn for any arbitrary patient inputs.
Outputs high-resolution base64 PNG images directly in memory.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_cleveland.csv"

# Cache raw dataset for population reference plots
_DATASET_CACHE: Optional[pd.DataFrame] = None


def get_dataset() -> pd.DataFrame:
    global _DATASET_CACHE
    if _DATASET_CACHE is None and DATA_PATH.exists():
        _DATASET_CACHE = pd.read_csv(DATA_PATH)
    return _DATASET_CACHE


def fig_to_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib figure to a base64 encoded PNG string and closes the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def set_dark_theme(ax: plt.Axes, fig: plt.Figure):
    """Sets a premium dark clinical aesthetic."""
    fig.patch.set_facecolor("#0b1329")
    ax.set_facecolor("#111c38")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#2a3b66")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#f8fafc")


def generate_feature_impact_plot(pipeline, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the exact Scikit-Learn Logistic Regression feature impact / log-odds contribution.
    Uses pipeline.named_steps['preprocessor'] and classifier.coef_.
    """
    from src.predict import FEATURE_NAMES

    # Build 1-row DataFrame
    row_dict = {col: patient_dict.get(col, 0) for col in FEATURE_NAMES}
    df_patient = pd.DataFrame([row_dict], columns=FEATURE_NAMES)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform patient
    X_trans = preprocessor.transform(df_patient)[0]
    coefs = classifier.coef_[0]
    feature_names = preprocessor.get_feature_names_out()

    contributions = X_trans * coefs

    # Clean readable names
    friendly_names = {
        "num__age": f"Age ({patient_dict.get('age', 0):.0f} yrs)",
        "num__trestbps": f"Resting BP ({patient_dict.get('trestbps', 0):.0f} mmHg)",
        "num__chol": f"Cholesterol ({patient_dict.get('chol', 0):.0f} mg/dL)",
        "num__thalach": f"Max Heart Rate ({patient_dict.get('thalach', 0):.0f} bpm)",
        "num__oldpeak": f"ST Depression ({patient_dict.get('oldpeak', 0):.1f} mm)",
        "cat__sex_1.0": "Male Biological Sex",
        "cat__sex_0.0": "Female Biological Sex",
        "cat__cp_1.0": "Typical Angina (cp=1)",
        "cat__cp_2.0": "Atypical Angina (cp=2)",
        "cat__cp_3.0": "Non-Anginal Pain (cp=3)",
        "cat__cp_4.0": "Asymptomatic Chest Pain (cp=4)",
        "cat__fbs_1.0": "Fasting Blood Sugar > 120",
        "cat__fbs_0.0": "Normal Blood Sugar <= 120",
        "cat__restecg_0.0": "Normal Resting ECG",
        "cat__restecg_1.0": "ECG ST-T Abnormality",
        "cat__restecg_2.0": "ECG LV Hypertrophy",
        "cat__exang_1.0": "Exercise Angina Present",
        "cat__exang_0.0": "No Exercise Angina",
        "cat__slope_1.0": "ST Slope: Upsloping",
        "cat__slope_2.0": "ST Slope: Flat (Ischemic)",
        "cat__slope_3.0": "ST Slope: Downsloping",
        "cat__ca_0.0": "Fluoroscopy: 0 Vessels Blocked",
        "cat__ca_1.0": "Fluoroscopy: 1 Vessel Calcified",
        "cat__ca_2.0": "Fluoroscopy: 2 Vessels Calcified",
        "cat__ca_3.0": "Fluoroscopy: 3 Vessels Calcified",
        "cat__thal_3.0": "Thallium: Normal Flow",
        "cat__thal_6.0": "Thallium: Fixed Defect",
        "cat__thal_7.0": "Thallium: Reversible Defect",
    }

    records = []
    for name, impact, val in zip(feature_names, contributions, X_trans):
        if abs(impact) > 0.001 or abs(val) > 0.001:
            clean_title = friendly_names.get(name, name)
            records.append((clean_title, impact))

    # Sort by absolute impact and take top 8
    records.sort(key=lambda x: abs(x[1]), reverse=True)
    top_records = records[:8]
    top_records.reverse()  # Bottom-up for horizontal bar

    names = [r[0] for r in top_records]
    vals = [r[1] for r in top_records]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    set_dark_theme(ax, fig)

    bars = ax.barh(names, vals, color=colors, height=0.55, edgecolor="none")
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.7)

    # Value labels
    for bar, val in zip(bars, vals):
        offset = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        sign = "+" if val > 0 else ""
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.2f}",
            va="center",
            ha=ha,
            color="#f1f5f9",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title("Scikit-Learn Logistic Regression: Feature Log-Odds Attribution", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Contribution to Log-Odds (Red = Pushes Toward CAD Risk | Green = Protective)", fontsize=10.5, labelpad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.25, color="#64748b")

    max_abs = max([abs(v) for v in vals] + [0.8])
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    base64_str = fig_to_base64(fig)
    return {
        "id": "feature_impact",
        "title": "Model Feature Contribution (Log-Odds Impact)",
        "library_used": "Scikit-Learn (pipeline coefficients) + Matplotlib",
        "reason": "Depicts exactly how the trained scikit-learn model's linear weights interpret this patient's individual markers toward disease risk vs health.",
        "image_base64": base64_str,
    }


def generate_biomarker_heatmap(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders a multi-marker cardiovascular severity heatmap using Seaborn (sns.heatmap).
    Compares the patient's vitals against clinical thresholds and population medians.
    """
    # Features and clinical norms
    # (name, patient_val, normal_ref, high_threshold, unit, is_inverse)
    metrics_info = [
        ("Resting BP", float(patient_dict.get("trestbps", 120)), 120.0, 140.0, "mmHg", False),
        ("Cholesterol", float(patient_dict.get("chol", 200)), 200.0, 240.0, "mg/dL", False),
        ("Max HR", float(patient_dict.get("thalach", 150)), 160.0, 115.0, "bpm", True),  # lower HR is worse
        ("ST Depression", float(patient_dict.get("oldpeak", 0)), 0.0, 1.5, "mm", False),
        ("Calcified Vessels", float(patient_dict.get("ca", 0)), 0.0, 1.0, "vessels", False),
    ]

    labels = []
    patient_vals_str = []
    ref_vals_str = []
    risk_scores = []

    for name, p_val, ref_val, high_val, unit, is_inv in metrics_info:
        labels.append(f"{name}\n({unit})")
        patient_vals_str.append(f"{p_val:.1f}")
        ref_vals_str.append(f"{ref_val:.1f}")

        # Compute normalized severity score (0.0 = safe, 1.0 = high risk)
        if is_inv:
            # e.g. HR: normal 160 -> score 0.1; 110 -> score 0.95
            score = np.clip((ref_val - p_val) / (ref_val - high_val), 0.05, 1.0)
        else:
            if ref_val == 0.0:
                score = np.clip(p_val / (high_val * 1.5), 0.05, 1.0)
            else:
                score = np.clip((p_val - ref_val * 0.8) / (high_val - ref_val * 0.8), 0.05, 1.0)
        risk_scores.append(score)

    data_matrix = np.array([risk_scores])

    # Annotations matrix
    annot_matrix = np.array([[f"Value: {v}\nRisk: {int(s*100)}%" for v, s in zip(patient_vals_str, risk_scores)]])

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.patch.set_facecolor("#0b1329")
    ax.set_facecolor("#111c38")

    # Seaborn heatmap
    sns.heatmap(
        data_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        cbar_kws={"label": "Clinical Risk Intensity (0% Safe -> 100% Critical)", "orientation": "horizontal", "pad": 0.28},
        xticklabels=labels,
        yticklabels=["Patient\nBiomarkers"],
        linewidths=2.5,
        linecolor="#0b1329",
        annot_kws={"size": 10.5, "weight": "bold", "color": "#ffffff"},
        ax=ax,
    )

    ax.tick_params(colors="#cbd5e1", labelsize=10.5)
    ax.set_title("Seaborn Clinical Biomarker Risk Heatmap", fontsize=13, fontweight="bold", color="#f8fafc", pad=12)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=9)
    cbar.ax.xaxis.label.set_color("#cbd5e1")

    base64_str = fig_to_base64(fig)
    return {
        "id": "biomarker_heatmap",
        "title": "Clinical Biomarker Multi-Metric Heatmap",
        "library_used": "Seaborn (sns.heatmap) + Matplotlib",
        "reason": "Depicts cardiovascular systemic stress across blood pressure, cholesterol, heart rate, and calcified vessels on an annotated thermal risk scale.",
        "image_base64": base64_str,
    }


def generate_ecg_ischemia_plot(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders an ECG exercise stress test ST-segment waveform tracing using Matplotlib.
    Depicts the exact ST depression (oldpeak mm) and slope geometry compared to normal baseline.
    """
    oldpeak = float(patient_dict.get("oldpeak", 0.0))
    slope = int(patient_dict.get("slope", 2))
    exang = int(patient_dict.get("exang", 0))

    slope_names = {1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"}
    slope_str = slope_names.get(slope, f"Slope {slope}")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    set_dark_theme(ax, fig)

    # Synthetic ECG wave points
    # P wave -> QRS -> J-point -> ST segment -> T wave
    x_baseline = np.linspace(0, 10, 400)
    # Baseline normal wave
    def ecg_wave(x, st_dep=0.0, st_slope=1):
        y = np.zeros_like(x)
        # P wave around 1-2
        y += 0.25 * np.exp(-((x - 1.5) ** 2) / 0.08)
        # Q dip around 2.8
        y -= 0.2 * np.exp(-((x - 2.8) ** 2) / 0.01)
        # R spike around 3.0
        y += 2.0 * np.exp(-((x - 3.0) ** 2) / 0.015)
        # S dip around 3.2
        y -= 0.5 * np.exp(-((x - 3.2) ** 2) / 0.015)

        # ST segment (between x=3.4 and x=5.5)
        st_mask = (x >= 3.4) & (x <= 5.5)
        t_norm = (x[st_mask] - 3.4) / (5.5 - 3.4)
        if st_slope == 1:  # upsloping
            st_y = -st_dep + st_dep * 0.7 * t_norm
        elif st_slope == 2:  # flat
            st_y = -st_dep * np.ones_like(t_norm)
        else:  # downsloping (severe)
            st_y = -st_dep - 0.4 * t_norm
        y[st_mask] = st_y

        # T wave around 6.5
        t_wave_amp = 0.45 if st_dep < 1.0 else -0.3  # inverted T if severe ischemia
        y += t_wave_amp * np.exp(-((x - 6.5) ** 2) / 0.4)
        return y

    y_normal = ecg_wave(x_baseline, st_dep=0.0, st_slope=1)
    y_patient = ecg_wave(x_baseline, st_dep=oldpeak, st_slope=slope)

    # Plot lines
    ax.plot(x_baseline, y_normal, label="Normal Rest Reference (0.0 mm)", color="#06b6d4", linestyle="--", linewidth=1.8, alpha=0.7)
    patient_color = "#ef4444" if oldpeak >= 1.5 else ("#f59e0b" if oldpeak >= 0.8 else "#10b981")
    ax.plot(x_baseline, y_patient, label=f"Patient Exercise Waveform (ST: -{oldpeak:.1f} mm, {slope_str})", color=patient_color, linewidth=2.4)

    # Fill ischemic gap
    st_idx = (x_baseline >= 3.4) & (x_baseline <= 5.5)
    ax.fill_between(x_baseline[st_idx], y_normal[st_idx], y_patient[st_idx], color="#ef4444", alpha=0.35, label="Ischemic ST Depression Zone")

    # Isoelectric line
    ax.axhline(0, color="#64748b", linestyle=":", linewidth=1.0)

    # Callout text box
    angina_text = "Exercise Angina: YES" if exang == 1 else "Exercise Angina: NO"
    ax.text(
        0.03,
        0.15,
        f"ST Depression (oldpeak): {oldpeak:.1f} mm\nST Slope Profile: {slope_str}\n{angina_text}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#f8fafc",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#475569", alpha=0.9),
    )

    ax.set_title("Matplotlib Exercise Stress ECG: ST-Segment Ischemia Profile", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Amplitude / ST Deviation (mm)", fontsize=10.5)
    ax.set_xlabel("Cardiac Cycle Progression (Time / Lead V5 Representation)", fontsize=10.5)
    ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", fontsize=9)
    ax.set_ylim(-max(oldpeak + 1.2, 1.8), 2.4)

    base64_str = fig_to_base64(fig)
    return {
        "id": "ecg_ischemia",
        "title": "Exercise ECG ST-Segment Ischemia Profile",
        "library_used": "Matplotlib (ECG waveform & ST deficit geometry)",
        "reason": "Depicts cardiac subendocardial ischemia through physical ST-segment depression depth and slope configuration during cardiac stress.",
        "image_base64": base64_str,
    }


def generate_age_thalach_kde_plot(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders an Age vs Max Heart Rate (thalach) cohort distribution using Seaborn & Matplotlib.
    Plots healthy vs CAD populations with the patient's exact coordinates and age-predicted maximum.
    """
    age = float(patient_dict.get("age", 55.0))
    thalach = float(patient_dict.get("thalach", 150.0))

    df = get_dataset()

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    set_dark_theme(ax, fig)

    if df is not None and "age" in df.columns and "thalach" in df.columns and "target" in df.columns:
        # Plot population points using seaborn
        sns.scatterplot(
            data=df,
            x="age",
            y="thalach",
            hue="target",
            palette={0: "#06b6d4", 1: "#f43f5e"},
            alpha=0.5,
            s=45,
            edgecolor="none",
            ax=ax,
        )

        # Plot KDE contour for Heart Disease patients
        cad_df = df[df["target"] == 1]
        if len(cad_df) > 10:
            sns.kdeplot(
                data=cad_df,
                x="age",
                y="thalach",
                color="#f43f5e",
                levels=4,
                alpha=0.45,
                linewidths=1.2,
                ax=ax,
            )

    # Astrand age-predicted max heart rate line: 220 - age
    age_range = np.linspace(28, 80, 100)
    max_hr_expected = 220 - age_range
    ax.plot(age_range, max_hr_expected, color="#facc15", linestyle="--", linewidth=1.5, label="Max Expected HR (220 - Age)")

    # Target patient indicator
    ax.scatter(
        [age],
        [thalach],
        color="#fbbf24",
        s=260,
        marker="*",
        edgecolor="#ffffff",
        linewidth=1.8,
        zorder=10,
        label=f"Current Patient (Age {age:.0f}, HR {thalach:.0f})",
    )

    # Annotation pointer
    ax.annotate(
        f"Patient\n({age:.0f}y, {thalach:.0f} bpm)",
        xy=(age, thalach),
        xytext=(age + 3, thalach + 10 if thalach < 170 else thalach - 22),
        color="#fef08a",
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(facecolor="#facc15", edgecolor="none", shrink=0.08, width=1.5, headwidth=6),
    )

    ax.set_title("Seaborn & Matplotlib: Cardiac Stress Reserve vs Patient Age", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Patient Age (Years)", fontsize=10.5)
    ax.set_ylabel("Maximum Heart Rate Achieved - thalach (bpm)", fontsize=10.5)

    # Re-build custom legend
    handles, labels = ax.get_legend_handles_labels()
    custom_labels = []
    for lbl in labels:
        if lbl == "0":
            custom_labels.append("Cleveland Cohort: Healthy (0)")
        elif lbl == "1":
            custom_labels.append("Cleveland Cohort: CAD (1)")
        else:
            custom_labels.append(lbl)
    ax.legend(handles, custom_labels, loc="lower left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", fontsize=8.5)

    base64_str = fig_to_base64(fig)
    return {
        "id": "age_thalach_kde",
        "title": "Cardiac Stress Space (Age vs Max Heart Rate)",
        "library_used": "Seaborn (sns.scatterplot + sns.kdeplot) + Matplotlib",
        "reason": "Depicts chronotropic competence and cardiovascular reserve relative to the Cleveland clinical cohort and age-predicted cardiac boundaries.",
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
    Intelligently selects ONLY 1 visualization that best depicts the patient's result,
    as required: 'not everytime it should give the all things, it should give only 1 of them which will depict the result properly'.

    If override_id is specified (e.g. from faculty UI toggle), renders that specific one.
    """
    oldpeak = float(patient_dict.get("oldpeak", 0.0))
    slope = int(patient_dict.get("slope", 2))
    exang = int(patient_dict.get("exang", 0))
    ca = float(patient_dict.get("ca", 0.0))
    chol = float(patient_dict.get("chol", 200.0))
    trestbps = float(patient_dict.get("trestbps", 120.0))
    thalach = float(patient_dict.get("thalach", 150.0))
    age = float(patient_dict.get("age", 55.0))

    # Determine automatic choice
    if override_id == "ecg_ischemia" or (
        override_id is None and (oldpeak >= 1.5 or (slope in [2, 3] and exang == 1))
    ):
        chosen = generate_ecg_ischemia_plot(patient_dict)
    elif override_id == "biomarker_heatmap" or (
        override_id is None and (ca >= 1.0 or chol >= 260.0 or trestbps >= 150.0)
    ):
        chosen = generate_biomarker_heatmap(patient_dict)
    elif override_id == "age_thalach_kde" or (
        override_id is None and (thalach < 120.0 or age >= 68.0)
    ):
        chosen = generate_age_thalach_kde_plot(patient_dict)
    else:
        # Default or override: feature log-odds contribution
        chosen = generate_feature_impact_plot(pipeline, patient_dict)

    return chosen
