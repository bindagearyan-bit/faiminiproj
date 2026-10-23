"""
predict.py

Inference and Custom Value Classification Script for the Cleveland Heart Disease Model.
Allows users to classify any patient values (not just historical data) using
the serialized scikit-learn Pipeline (imputation + scaling + encoding + logistic regression).

Usage Modes:
  1. Interactive CLI (prompts user for each feature with guidance):
       python src/predict.py
  2. Command-Line Arguments:
       python src/predict.py --age 58 --sex 1 --cp 4 --trestbps 140 --chol 260 --fbs 0 --restecg 1 --thalach 130 --exang 1 --oldpeak 2.2 --slope 2 --ca 1 --thal 7
  3. Pre-configured Demo Scenarios (Low-Risk, Moderate-Risk, High-Risk):
       python src/predict.py --demo
  4. JSON File or Inline JSON:
       python src/predict.py --json "{\"age\": 60, \"sex\": 1, \"cp\": 4, ...}"
  5. Batch CSV:
       python src/predict.py --csv path/to/patients.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# The 13 clinical predictor features expected by the trained pipeline
FEATURE_NAMES: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Clinical metadata, bounds, descriptions, and default values for interactive input
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {
        "name": "Age",
        "type": float,
        "default": 55.0,
        "min": 18.0,
        "max": 120.0,
        "unit": "years",
        "description": "Patient age in years",
        "help": "Enter patient age in years (e.g., 25 - 90)",
    },
    "sex": {
        "name": "Sex",
        "type": int,
        "default": 1,
        "choices": [0, 1],
        "unit": "binary",
        "description": "Biological sex (0 = Female, 1 = Male)",
        "help": "0 = Female, 1 = Male",
    },
    "cp": {
        "name": "Chest Pain Type (cp)",
        "type": int,
        "default": 4,
        "choices": [1, 2, 3, 4],
        "unit": "category",
        "description": "1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic",
        "help": "1: Typical angina | 2: Atypical angina | 3: Non-anginal pain | 4: Asymptomatic",
    },
    "trestbps": {
        "name": "Resting Blood Pressure (trestbps)",
        "type": float,
        "default": 130.0,
        "min": 70.0,
        "max": 250.0,
        "unit": "mm Hg",
        "description": "Resting blood pressure in mm Hg on admission to the hospital",
        "help": "Normal < 120, Pre-hypertension 120-139, Hypertension >= 140",
    },
    "chol": {
        "name": "Serum Cholesterol (chol)",
        "type": float,
        "default": 240.0,
        "min": 100.0,
        "max": 600.0,
        "unit": "mg/dl",
        "description": "Serum cholesterol in mg/dl",
        "help": "Desirable < 200, Borderline 200-239, High >= 240",
    },
    "fbs": {
        "name": "Fasting Blood Sugar > 120 mg/dl (fbs)",
        "type": int,
        "default": 0,
        "choices": [0, 1],
        "unit": "binary",
        "description": "Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)",
        "help": "0: <= 120 mg/dl (normal) | 1: > 120 mg/dl (elevated/diabetic)",
    },
    "restecg": {
        "name": "Resting ECG Results (restecg)",
        "type": int,
        "default": 0,
        "choices": [0, 1, 2],
        "unit": "category",
        "description": "0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy",
        "help": "0: Normal | 1: ST-T wave abnormality | 2: LV hypertrophy (Estes' criteria)",
    },
    "thalach": {
        "name": "Maximum Heart Rate Achieved (thalach)",
        "type": float,
        "default": 150.0,
        "min": 50.0,
        "max": 230.0,
        "unit": "bpm",
        "description": "Peak heart rate reached during exercise stress testing",
        "help": "Typically 70 - 200 bpm (higher values generally indicate better cardiac fitness)",
    },
    "exang": {
        "name": "Exercise-Induced Angina (exang)",
        "type": int,
        "default": 0,
        "choices": [0, 1],
        "unit": "binary",
        "description": "Exercise-induced angina (1 = Yes, 0 = No)",
        "help": "0: No | 1: Yes (anginal chest pain triggered by exercise)",
    },
    "oldpeak": {
        "name": "ST Depression (oldpeak)",
        "type": float,
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
        "unit": "mm",
        "description": "ST depression induced by exercise relative to rest",
        "help": "0.0 = Normal (no depression) | > 1.0 indicates cardiac ischemia",
    },
    "slope": {
        "name": "Slope of Peak Exercise ST (slope)",
        "type": int,
        "default": 2,
        "choices": [1, 2, 3],
        "unit": "category",
        "description": "1 = Upsloping, 2 = Flat, 3 = Downsloping",
        "help": "1: Upsloping (normal) | 2: Flat (ischemic) | 3: Downsloping (severe ischemia)",
    },
    "ca": {
        "name": "Major Vessels Colored by Fluoroscopy (ca)",
        "type": float,
        "default": 0.0,
        "choices": [0.0, 1.0, 2.0, 3.0],
        "unit": "count",
        "description": "Number of major blood vessels (0-3) colored by fluoroscopy",
        "help": "0: No blocked vessels detected | 1, 2, 3: Number of visible coronary vessels",
    },
    "thal": {
        "name": "Thalassemia Perfusion (thal)",
        "type": float,
        "default": 3.0,
        "choices": [3.0, 6.0, 7.0],
        "unit": "category",
        "description": "3.0 = Normal, 6.0 = Fixed defect, 7.0 = Reversible defect",
        "help": "3: Normal flow | 6: Fixed defect (prior infarct) | 7: Reversible defect (active ischemia)",
    },
}

# Predefined educational presets for fast testing
PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "2": {
        "title": "High-Risk Cardiac Patient (Ischemia & Multi-vessel Disease)",
        "description": "67-year-old male with asymptomatic chest pain, hypertension, ST depression, and 2 blocked vessels",
        "data": {
            "age": 67.0,
            "sex": 1,
            "cp": 4,
            "trestbps": 160.0,
            "chol": 286.0,
            "fbs": 0,
            "restecg": 2,
            "thalach": 108.0,
            "exang": 1,
            "oldpeak": 2.6,
            "slope": 2,
            "ca": 2.0,
            "thal": 7.0,
        },
    },
    "3": {
        "title": "Low-Risk / Healthy Cardiac Profile",
        "description": "37-year-old female, non-anginal pain, normal ECG, excellent peak heart rate, 0 blocked vessels",
        "data": {
            "age": 37.0,
            "sex": 0,
            "cp": 3,
            "trestbps": 120.0,
            "chol": 195.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 178.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0.0,
            "thal": 3.0,
        },
    },
    "4": {
        "title": "Moderate / Borderline Risk Profile",
        "description": "54-year-old male, atypical angina, mild resting hypertension, moderate ST depression",
        "data": {
            "age": 54.0,
            "sex": 1,
            "cp": 2,
            "trestbps": 138.0,
            "chol": 235.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 2,
            "ca": 0.0,
            "thal": 3.0,
        },
    },
}


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_model_pipeline(pipeline_path: Optional[Path] = None):
    """
    Loads the trained scikit-learn Pipeline from disk.
    If the artifact is missing, runs train_model.py to generate it.
    """
    project_root = get_project_root()
    if pipeline_path is None:
        pipeline_path = project_root / "models" / "heart_disease_pipeline.joblib"

    if not pipeline_path.exists():
        print(f"[INFO] Pipeline artifact not found at {pipeline_path}. Training model now...")
        from src.train_model import train_and_evaluate
        train_and_evaluate()

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Could not load or train model pipeline at: {pipeline_path.resolve()}"
        )

    pipeline = joblib.load(pipeline_path)
    return pipeline


def predict_patient(
    patient_data: Dict[str, Any],
    pipeline=None
) -> Dict[str, Any]:
    """
    Classifies an arbitrary patient profile using the trained machine learning pipeline.

    Args:
        patient_data: Dictionary mapping the 13 feature names to their numeric/categorical values.
        pipeline: Optional pre-loaded scikit-learn Pipeline. If None, loaded from disk.

    Returns:
        Structured result dictionary containing:
          - prediction: int (0 = absence, 1 = presence)
          - label: str description
          - prob_disease: float (0.0 to 1.0)
          - prob_no_disease: float (0.0 to 1.0)
          - risk_tier: str ('Low Risk', 'Moderate Risk', 'High Risk')
          - input_features: validated dictionary
          - key_factors: list of clinical indicators noted in this patient
    """
    if pipeline is None:
        pipeline = load_model_pipeline()

    # Build single-row DataFrame with strict column order
    row_dict: Dict[str, Any] = {}
    for col in FEATURE_NAMES:
        if col not in patient_data:
            meta = FEATURE_METADATA.get(col, {})
            row_dict[col] = meta.get("default", 0)
        else:
            val = patient_data[col]
            # Convert numeric strings if passed
            try:
                row_dict[col] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                row_dict[col] = val

    df = pd.DataFrame([row_dict], columns=FEATURE_NAMES)

    # Scikit-learn Pipeline handles imputation, scaling, and one-hot encoding internally
    raw_pred = pipeline.predict(df)[0]
    prediction = int(raw_pred)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df)[0]
        prob_no_disease = float(probabilities[0])
        prob_disease = float(probabilities[1])
    else:
        prob_disease = 1.0 if prediction == 1 else 0.0
        prob_no_disease = 1.0 - prob_disease

    # Risk tier determination
    if prob_disease >= 0.70:
        risk_tier = "High Risk"
    elif prob_disease >= 0.35:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "Low Risk"

    # Analyze key indicators in simple terms for AIML presentation
    key_factors = []
    if row_dict["cp"] == 4:
        key_factors.append("Silent / Asymptomatic Chest Pain ('cp'=4, strongest statistical heart disease indicator in dataset)")
    elif row_dict["cp"] in [1, 2]:
        key_factors.append("Chest pain or heaviness reported ('cp'=1 or 2)")

    if row_dict["ca"] > 0:
        key_factors.append(f"{int(row_dict['ca'])} Blocked Blood Vessel(s) detected (ca={int(row_dict['ca'])})")
    if row_dict["oldpeak"] >= 1.5:
        key_factors.append(f"High ECG Stress test score ({row_dict['oldpeak']}, shows heart muscle strain)")
    if row_dict["thal"] == 7.0:
        key_factors.append("Blood flow test shows severe blockage in coronary arteries (thal=7.0)")
    elif row_dict["thal"] == 6.0:
        key_factors.append("Blood flow test shows partial blockage or prior damage (thal=6.0)")
    if row_dict["exang"] == 1:
        key_factors.append("Chest pain triggered by exercise (exang=1)")
    if row_dict["trestbps"] >= 140:
        key_factors.append(f"High Blood Pressure ({row_dict['trestbps']} mm Hg, normal is ~120)")
    if row_dict["chol"] >= 240:
        key_factors.append(f"High Cholesterol ({row_dict['chol']} mg/dl, desirable is < 200)")
    if row_dict["thalach"] < 120:
        key_factors.append(f"Low Maximum Heart Rate ({row_dict['thalach']} bpm, heart cannot pump adequately under load)")

    label = "Heart Disease Detected (Class 1)" if prediction == 1 else "Healthy / No Heart Disease (Class 0)"

    return {
        "prediction": prediction,
        "label": label,
        "prob_disease": prob_disease,
        "prob_no_disease": prob_no_disease,
        "risk_tier": risk_tier,
        "input_features": row_dict,
        "key_factors": key_factors,
    }


def format_prediction_report(result: Dict[str, Any], title: str = "PATIENT CLASSIFICATION REPORT") -> str:
    """
    Renders a clean, high-visibility clinical prediction report in terminal-friendly format.
    """
    inputs = result["input_features"]
    pred = result["prediction"]
    prob_disease = result["prob_disease"]
    prob_no_disease = result["prob_no_disease"]
    risk_tier = result["risk_tier"]
    factors = result["key_factors"]

    # Decode categories for human readability
    sex_str = "Male (1)" if inputs["sex"] == 1 else "Female (0)"
    cp_map = {1: "Typical Angina (1)", 2: "Atypical Angina (2)", 3: "Non-Anginal Pain (3)", 4: "Asymptomatic (4)"}
    cp_str = cp_map.get(int(inputs["cp"]), str(inputs["cp"]))
    fbs_str = "> 120 mg/dl [True] (1)" if inputs["fbs"] == 1 else "<= 120 mg/dl [False] (0)"
    restecg_map = {0: "Normal (0)", 1: "ST-T Wave Abnormality (1)", 2: "LV Hypertrophy (2)"}
    restecg_str = restecg_map.get(int(inputs["restecg"]), str(inputs["restecg"]))
    exang_str = "Yes (1)" if inputs["exang"] == 1 else "No (0)"
    slope_map = {1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"}
    slope_str = slope_map.get(int(inputs["slope"]), str(inputs["slope"]))
    thal_map = {3: "Normal (3.0)", 6: "Fixed Defect (6.0)", 7: "Reversible Defect (7.0)"}
    thal_str = thal_map.get(int(inputs["thal"]), str(inputs["thal"]))

    border = "=" * 76
    sub_border = "-" * 76

    lines = [
        border,
        f"  {title}",
        border,
        "",
        "  [1] PATIENT CLINICAL ATTRIBUTES RECEIVED:",
        f"      * Age                      : {inputs['age']} years",
        f"      * Sex                      : {sex_str}",
        f"      * Chest Pain Type (cp)     : {cp_str}",
        f"      * Resting BP (trestbps)    : {inputs['trestbps']} mm Hg",
        f"      * Cholesterol (chol)       : {inputs['chol']} mg/dl",
        f"      * Fasting Blood Sugar (fbs): {fbs_str}",
        f"      * Resting ECG (restecg)    : {restecg_str}",
        f"      * Max Heart Rate (thalach) : {inputs['thalach']} bpm",
        f"      * Exercise Angina (exang)  : {exang_str}",
        f"      * ST Depression (oldpeak)  : {inputs['oldpeak']} mm",
        f"      * ST Segment Slope (slope) : {slope_str}",
        f"      * Major Vessels (ca)       : {int(inputs['ca'])} vessels colored",
        f"      * Thallium Perfusion (thal): {thal_str}",
        "",
        sub_border,
        "  [2] MODEL CLASSIFICATION RESULT:",
    ]

    if pred == 1:
        lines.extend([
            f"      DIAGNOSIS CLASSIFICATION : [ POSITIVE: HEART DISEASE DETECTED (Class 1) ]",
            f"      ASSESSED RISK TIER       : [ {risk_tier.upper()} ]",
            f"      DISEASE PROBABILITY      : {prob_disease * 100:.2f}%",
            f"      ABSENCE PROBABILITY      : {prob_no_disease * 100:.2f}%",
        ])
    else:
        lines.extend([
            f"      DIAGNOSIS CLASSIFICATION : [ NEGATIVE: NO HEART DISEASE DETECTED (Class 0) ]",
            f"      ASSESSED RISK TIER       : [ {risk_tier.upper()} ]",
            f"      ABSENCE PROBABILITY      : {prob_no_disease * 100:.2f}%",
            f"      DISEASE PROBABILITY      : {prob_disease * 100:.2f}%",
        ])

    lines.extend([
        "",
        sub_border,
        "  [3] CLINICAL RISK FACTOR ANALYSIS:",
    ])

    if factors:
        for factor in factors:
            lines.append(f"      - {factor}")
    else:
        lines.append("      - No high-risk cardiac flags identified in the provided features.")

    lines.extend([
        "",
        sub_border,
        "  * EDUCATIONAL NOTICE: This classification is generated by a statistical model",
        "    trained on the 1989 UCI Cleveland Dataset. It is NOT a medical diagnosis.",
        border,
    ])

    return "\n".join(lines)


def prompt_user_for_patient() -> Dict[str, Any]:
    """
    Guides the user through entering custom clinical values via terminal prompts,
    with clear explanations, default values, and input validation.
    """
    print("\n" + "=" * 70)
    print("  HEART DISEASE CLASSIFIER - INTERACTIVE VALUE INPUT")
    print("=" * 70)
    print("Choose an option:")
    print("  [1] Enter your own custom patient values step-by-step")
    print("  [2] Load High-Risk Patient Preset (CAD positive)")
    print("  [3] Load Low-Risk / Healthy Patient Preset (CAD negative)")
    print("  [4] Load Moderate / Borderline Patient Preset")
    print("-" * 70)

    choice = input("Select option [1/2/3/4] (Default: 1): ").strip()
    if choice in PRESET_PROFILES:
        preset = PRESET_PROFILES[choice]
        print(f"\n[SELECTED PRESET] {preset['title']}")
        print(f"Details: {preset['description']}\n")
        return preset["data"].copy()

    print("\nPlease enter values for the 13 clinical variables below.")
    print("Tip: Press [Enter] without typing to accept the suggested default value.")
    print("-" * 70)

    custom_data: Dict[str, Any] = {}
    for feature_name in FEATURE_NAMES:
        meta = FEATURE_METADATA[feature_name]
        default = meta["default"]
        val_type = meta["type"]
        help_text = meta.get("help", "")

        while True:
            prompt_str = f"-> {meta['name']}\n   ({help_text})\n   [Default: {default}]: "
            user_input = input(prompt_str).strip()

            if not user_input:
                custom_data[feature_name] = default
                break

            try:
                if val_type == int:
                    parsed_val = int(user_input)
                else:
                    parsed_val = float(user_input)

                # Check choices if restricted
                if "choices" in meta and parsed_val not in meta["choices"]:
                    print(f"   [!] Invalid value '{parsed_val}'. Permitted choices: {meta['choices']}")
                    continue

                # Check range min/max
                if "min" in meta and parsed_val < meta["min"]:
                    print(f"   [!] Value must be >= {meta['min']}.")
                    continue
                if "max" in meta and parsed_val > meta["max"]:
                    print(f"   [!] Value must be <= {meta['max']}.")
                    continue

                custom_data[feature_name] = parsed_val
                break
            except ValueError:
                print(f"   [!] Could not parse '{user_input}' as a valid number. Please retry.")

        print()

    return custom_data


def run_demo(pipeline=None):
    """Runs prediction across all 3 standard demo patient profiles."""
    if pipeline is None:
        pipeline = load_model_pipeline()

    print("\n" + "#" * 76)
    print("  DEMONSTRATION: CLASSIFYING ARBITRARY PATIENT PROFILES")
    print("#" * 76)

    for key, preset in PRESET_PROFILES.items():
        print(f"\n>>> Running Scenario {key}: {preset['title']}")
        result = predict_patient(preset["data"], pipeline=pipeline)
        report = format_prediction_report(result, title=f"DEMO SCENARIO {key}: {preset['title']}")
        print(report)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify any custom values using the trained Heart Disease ML Pipeline."
    )
    # 13 clinical variables
    parser.add_argument("--age", type=float, help="Age in years (e.g. 58)")
    parser.add_argument("--sex", type=int, choices=[0, 1], help="Sex (0 = female, 1 = male)")
    parser.add_argument("--cp", type=int, choices=[1, 2, 3, 4], help="Chest pain type (1-4)")
    parser.add_argument("--trestbps", type=float, help="Resting blood pressure in mm Hg (e.g. 140)")
    parser.add_argument("--chol", type=float, help="Serum cholesterol in mg/dl (e.g. 260)")
    parser.add_argument("--fbs", type=int, choices=[0, 1], help="Fasting blood sugar > 120 mg/dl (0 or 1)")
    parser.add_argument("--restecg", type=int, choices=[0, 1, 2], help="Resting ECG results (0, 1, or 2)")
    parser.add_argument("--thalach", type=float, help="Maximum heart rate achieved in bpm (e.g. 150)")
    parser.add_argument("--exang", type=int, choices=[0, 1], help="Exercise-induced angina (0 or 1)")
    parser.add_argument("--oldpeak", type=float, help="ST depression induced by exercise (e.g. 1.5)")
    parser.add_argument("--slope", type=int, choices=[1, 2, 3], help="Slope of peak exercise ST segment (1, 2, or 3)")
    parser.add_argument("--ca", type=float, choices=[0.0, 1.0, 2.0, 3.0], help="Major vessels colored by fluoroscopy (0-3)")
    parser.add_argument("--thal", type=float, choices=[3.0, 6.0, 7.0], help="Thalassemia (3.0=normal, 6.0=fixed, 7.0=reversible)")

    # Execution flags
    parser.add_argument("--interactive", action="store_true", help="Launch interactive step-by-step terminal prompt")
    parser.add_argument("--demo", action="store_true", help="Run classifications on preconfigured demo patients")
    parser.add_argument("--json", type=str, help="JSON string or path to JSON file containing patient attributes")
    parser.add_argument("--csv", type=str, help="Path to CSV file with patient records for batch classification")

    return parser.parse_args()


def main():
    args = parse_arguments()
    pipeline = load_model_pipeline()

    # 1. Demo mode
    if args.demo:
        run_demo(pipeline=pipeline)
        return

    # 2. JSON input mode
    if args.json:
        json_path = Path(args.json)
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
        else:
            patient_data = json.loads(args.json)

        result = predict_patient(patient_data, pipeline=pipeline)
        print(format_prediction_report(result, title="JSON INPUT CLASSIFICATION"))
        return

    # 3. CSV batch mode
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] CSV file not found: {csv_path.resolve()}")
            sys.exit(1)

        batch_df = pd.read_csv(csv_path)
        print(f"\n[INFO] Loaded {len(batch_df)} patient records from {csv_path.name}")
        predictions = pipeline.predict(batch_df[FEATURE_NAMES])
        probabilities = pipeline.predict_proba(batch_df[FEATURE_NAMES])

        batch_df["predicted_target"] = predictions
        batch_df["predicted_label"] = [
            "Disease (1)" if p == 1 else "No Disease (0)" for p in predictions
        ]
        batch_df["prob_heart_disease"] = probabilities[:, 1].round(4)
        batch_df["prob_no_disease"] = probabilities[:, 0].round(4)

        output_csv = csv_path.parent / f"{csv_path.stem}_predictions.csv"
        batch_df.to_csv(output_csv, index=False)
        print(f"[SUCCESS] Saved batch predictions to: {output_csv.resolve()}")
        print(batch_df[["predicted_target", "predicted_label", "prob_heart_disease"]].head(10))
        return

    # 4. CLI arguments mode (if any feature argument was provided)
    cli_features = {
        k: getattr(args, k)
        for k in FEATURE_NAMES
        if getattr(args, k) is not None
    }

    if cli_features:
        # Fill missing features with default metadata if not all 13 are specified
        patient_data = {}
        for feature_name in FEATURE_NAMES:
            if feature_name in cli_features:
                patient_data[feature_name] = cli_features[feature_name]
            else:
                patient_data[feature_name] = FEATURE_METADATA[feature_name]["default"]

        result = predict_patient(patient_data, pipeline=pipeline)
        print(format_prediction_report(result, title="CUSTOM PATIENT VALUES CLASSIFICATION"))
        return

    # 5. Default: Interactive mode
    patient_data = prompt_user_for_patient()
    result = predict_patient(patient_data, pipeline=pipeline)
    print("\n" + format_prediction_report(result, title="CUSTOM VALUE CLASSIFICATION RESULT"))


if __name__ == "__main__":
    main()
