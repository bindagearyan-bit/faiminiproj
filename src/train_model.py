"""
train_model.py

Model training and evaluation pipeline for the Cleveland Heart Disease dataset.
Adheres strictly to best ML practices:
  - Prevents data leakage by encapsulating preprocessing in scikit-learn Pipelines.
  - Predictors X strictly exclude 'target' and 'target_original'.
  - Stratified 75/25 train/test split (random_state=42).
  - Handles categorical columns (most frequent impute + OneHotEncode).
  - Handles numeric columns (median impute + StandardScaler).
  - Evaluates DummyClassifier baseline, LogisticRegression, and RandomForestClassifier
    using 5-fold StratifiedKFold cross-validation F1 on the training set only.
  - Selects the top-performing non-baseline model based purely on training CV F1.
  - Fits the chosen pipeline on the full training set and evaluates ONCE on the test set.
  - Saves:
      * Confusion matrix image -> outputs/figures/confusion_matrix.png
      * Metrics CSVs -> outputs/metrics/model_evaluation_metrics.csv
      * Fitted pipeline -> models/heart_disease_pipeline.joblib
"""

from pathlib import Path
import sys
import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_and_evaluate():
    # 1. Project-root-relative paths
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "heart_disease_cleveland.csv"
    models_dir = project_root / "models"
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"

    # Ensure output directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HEART DISEASE CLASSIFICATION - MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 2. Check dataset presence
    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found at: {data_path.resolve()}")
        print("Please run 'python src/download_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nLoaded dataset from: {data_path.relative_to(project_root)}")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Target isolation & strict leakage prevention
    target_columns = [col for col in ["target", "target_original"] if col in df.columns]
    if "target" not in df.columns:
        print(f"[ERROR] Required 'target' column missing. Found columns: {list(df.columns)}")
        sys.exit(1)

    X = df.drop(columns=target_columns)
    y = df["target"]

    # Verify no target leakage in feature matrix X
    assert "target" not in X.columns, "CRITICAL ERROR: 'target' found in predictors X!"
    assert "target_original" not in X.columns, "CRITICAL ERROR: 'target_original' found in predictors X!"
    print(f"\nPredictor columns ({len(X.columns)} features): {list(X.columns)}")
    print(f"Target column: 'target' (Excluded all targets from X to prevent leakage)")

    # 4. Stratified Train/Test Split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain set shape: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set shape:  {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(f"Train class distribution:\n{y_train.value_counts().to_dict()}")
    print(f"Test class distribution:\n{y_test.value_counts().to_dict()}")

    # 5. Column Identification & Preprocessing Setup
    # Treat these columns as categorical when present
    cat_candidates = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    cat_cols = [c for c in cat_candidates if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numeric features ({len(num_cols)}):     {num_cols}")

    # Build scikit-learn Pipeline transformations
    # Categorical: impute with most frequent value, then one-hot encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Numeric: impute with median, then standardize
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # 6. Cross-Validation Sanity Check
    class_counts = y_train.value_counts()
    n_splits = 5
    if len(class_counts) < 2:
        raise ValueError(
            f"[FATAL] Cross-validation impossible: Training set contains only 1 class ({class_counts.to_dict()}). "
            f"Binary classification requires 2 classes."
        )
    if class_counts.min() < n_splits:
        raise ValueError(
            f"[FATAL] Cross-validation impossible: Smallest class in training set has {class_counts.min()} samples, "
            f"which is less than the requested {n_splits} folds. Class counts: {class_counts.to_dict()}."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 7. Model Candidates Comparison via 5-Fold Stratified CV on Training Set
    candidate_models = {
        "Dummy Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100)
    }

    print("\n" + "-" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET (Metric: F1 Score)")
    print("-" * 70)

    cv_results = []
    for model_name, model_instance in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model_instance)
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        mean_f1 = scores.mean()
        std_f1 = scores.std()
        is_baseline = "Dummy" in model_name

        cv_results.append({
            "Model": model_name,
            "Mean_CV_F1": round(mean_f1, 4),
            "Std_CV_F1": round(std_f1, 4),
            "Is_Baseline": is_baseline
        })

        print(f"  {model_name:<32}: Mean F1 = {mean_f1:.4f} (+/- {std_f1:.4f})")

    cv_df = pd.DataFrame(cv_results)
    cv_csv_path = metrics_dir / "cv_model_comparison.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"\n[SAVED] Cross-validation comparison table -> {cv_csv_path.relative_to(project_root)}")

    # 8. Selection of Winning Non-Baseline Model
    non_baseline_results = [r for r in cv_results if not r["Is_Baseline"]]
    winning_info = max(non_baseline_results, key=lambda x: x["Mean_CV_F1"])
    winning_model_name = winning_info["Model"]
    winning_model_obj = candidate_models[winning_model_name]

    print("\n" + "=" * 70)
    print(f"WINNING MODEL SELECTED: {winning_model_name}")
    print(f"Selection Criterion: Highest Mean 5-Fold CV F1 on Training Data ({winning_info['Mean_CV_F1']:.4f})")
    print("(Note: Test set was NOT used to choose the model, avoiding data snooping.)")
    print("=" * 70)

    # 9. Fit Winning Pipeline on ENTIRE Training Set
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", winning_model_obj)
    ])
    final_pipeline.fit(X_train, y_train)

    # 10. Single Evaluation on Held-Out Test Set
    y_pred = final_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred, target_names=["0 (No Disease)", "1 (Disease)"]
    )

    print("\n" + "-" * 70)
    print(f"FINAL EVALUATION ON HELD-OUT TEST SET (N={len(y_test)} patients)")
    print("-" * 70)
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall   : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(f"  [[TN={cm[0, 0]:>2}, FP={cm[0, 1]:>2}]")
    print(f"   [FN={cm[1, 0]:>2}, TP={cm[1, 1]:>2}]]")

    print("\nDetailed Classification Report:")
    print(report_text)

    # 11. Save Confusion Matrix Figure
    cm_fig_path = figures_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["0 (No Disease)", "1 (Disease)"]
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Test Confusion Matrix - {winning_model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrix image -> {cm_fig_path.relative_to(project_root)}")

    # 12. Save Test Metrics CSV
    metrics_summary = pd.DataFrame([{
        "winning_model": winning_model_name,
        "cv_mean_f1_train": winning_info["Mean_CV_F1"],
        "cv_std_f1_train": winning_info["Std_CV_F1"],
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "test_true_negatives": int(cm[0, 0]),
        "test_false_positives": int(cm[0, 1]),
        "test_false_negatives": int(cm[1, 0]),
        "test_true_positives": int(cm[1, 1]),
        "test_samples": len(y_test)
    }])
    metrics_csv_path = metrics_dir / "model_evaluation_metrics.csv"
    metrics_summary.to_csv(metrics_csv_path, index=False)
    print(f"[SAVED] Test metrics summary -> {metrics_csv_path.relative_to(project_root)}")

    # 13. Save Fitted Winning Pipeline
    pipeline_path = models_dir / "heart_disease_pipeline.joblib"
    joblib.dump(final_pipeline, pipeline_path)
    print(f"[SAVED] Fitted pipeline artifact -> {pipeline_path.relative_to(project_root)}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
