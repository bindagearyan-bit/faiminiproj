"""
eda.py

Exploratory Data Analysis (EDA) for the Cleveland Heart Disease dataset.
Reads data/raw/heart_disease_cleveland.csv using project-root-relative paths.
Prints:
  - Dataset shape
  - Missing-value counts per column
  - Target class distribution
  - Descriptive summary statistics
Generates and saves at least 4 exploratory figures in outputs/figures/:
  1. target_counts.png: Distribution of binary heart disease diagnosis
  2. age_distribution.png: Patient age distribution with mean/median markers
  3. numeric_feature_by_target.png: Comparison of max heart rate (thalach) by target
  4. correlation_heatmap.png: Correlation matrix heatmap across numeric features
Handles missing or unexpected column names gracefully.
"""

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda():
    # Resolve paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "heart_disease_cleveland.csv"
    figures_dir = project_root / "outputs" / "figures"

    # Automatically create the output directory
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EXPLORATORY DATA ANALYSIS (EDA) - HEART DISEASE DATASET")
    print("=" * 65)

    # Validate that data file exists
    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found at: {data_path.resolve()}")
        print("Please run 'python src/download_data.py' before running this script.")
        sys.exit(1)

    # Read local dataset
    df = pd.read_csv(data_path)
    print(f"\nLoaded dataset from: {data_path.relative_to(project_root)}")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Missing Values
    print("\n--- Missing Value Counts Per Column ---")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("No missing values found across any columns.")
    else:
        for col, cnt in missing_cols.items():
            pct = (cnt / len(df)) * 100
            print(f"  {col:<12}: {cnt} missing ({pct:.2f}%)")

    # 2. Target Counts
    print("\n--- Target Class Distribution ---")
    if "target" in df.columns:
        counts = df["target"].value_counts().sort_index()
        for val, cnt in counts.items():
            label = "Absence of Heart Disease (0)" if val == 0 else "Presence of Heart Disease (1)"
            pct = (cnt / len(df)) * 100
            print(f"  {label:<35}: {cnt:>3} ({pct:.1f}%)")
    else:
        print(f"[WARNING] 'target' column not found! Found columns: {list(df.columns)}")

    # 3. Summary Statistics
    print("\n--- Descriptive Summary Statistics (Numeric Features) ---")
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]].to_string())

    # Set seaborn theme for beautiful, clean visualizations
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # -------------------------------------------------------------
    # Chart 1: Target Counts
    # -------------------------------------------------------------
    chart1_path = figures_dir / "target_counts.png"
    if "target" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        palette = ["#2b5c8f", "#d95f02"]
        target_counts = df["target"].value_counts().sort_index()
        bars = ax.bar(["0 (No Disease)", "1 (Heart Disease)"], target_counts.values, color=palette, width=0.55)

        for bar in bars:
            height = bar.get_height()
            pct = (height / len(df)) * 100
            ax.annotate(
                f"{height} ({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        ax.set_title("Heart Disease Target Distribution (Cleveland Subset)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Patient Count", fontsize=11)
        ax.set_xlabel("Diagnosis Class", fontsize=11)
        ax.set_ylim(0, max(target_counts.values) * 1.15)
        plt.tight_layout()
        plt.savefig(chart1_path, dpi=300)
        plt.close()
        print(f"\n[SAVED] Target distribution chart: {chart1_path.relative_to(project_root)}")
    else:
        print(f"[SKIP] Chart 1 skipped: 'target' not found. Found columns: {list(df.columns)}")

    # -------------------------------------------------------------
    # Chart 2: Age Distribution
    # -------------------------------------------------------------
    chart2_path = figures_dir / "age_distribution.png"
    if "age" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["age"], kde=True, color="#1b9e77", bins=15, ax=ax)
        mean_age = df["age"].mean()
        median_age = df["age"].median()

        ax.axvline(mean_age, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: {mean_age:.1f} yrs")
        ax.axvline(median_age, color="#7570b3", linestyle="-.", linewidth=2, label=f"Median: {median_age:.1f} yrs")

        ax.set_title("Patient Age Distribution", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Age (Years)", fontsize=11)
        ax.set_ylabel("Patient Count", fontsize=11)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(chart2_path, dpi=300)
        plt.close()
        print(f"[SAVED] Age distribution chart: {chart2_path.relative_to(project_root)}")
    else:
        print(f"[SKIP] Chart 2 skipped: 'age' not found. Found columns: {list(df.columns)}")

    # -------------------------------------------------------------
    # Chart 3: Numeric Feature by Target (e.g. thalach - max heart rate)
    # -------------------------------------------------------------
    chart3_path = figures_dir / "numeric_feature_by_target.png"
    # Identify candidate numeric feature: prefer 'thalach', fallback to others
    candidate_numeric = ["thalach", "trestbps", "chol", "age", "oldpeak"]
    chosen_numeric = None
    for col in candidate_numeric:
        if col in df.columns:
            chosen_numeric = col
            break

    if chosen_numeric and "target" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = {0: "0 (No Disease)", 1: "1 (Heart Disease)"}
        plot_df = df.copy()
        plot_df["target_label"] = plot_df["target"].map(labels)

        palette = {"0 (No Disease)": "#2b5c8f", "1 (Heart Disease)": "#d95f02"}
        sns.boxplot(
            data=plot_df,
            x="target_label",
            y=chosen_numeric,
            hue="target_label",
            legend=False,
            palette=palette,
            width=0.45,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
            ax=ax
        )
        sns.stripplot(
            data=plot_df,
            x="target_label",
            y=chosen_numeric,
            color="black",
            alpha=0.25,
            jitter=0.2,
            size=4,
            ax=ax
        )

        display_name = {
            "thalach": "Max Heart Rate Achieved (thalach, bpm)",
            "trestbps": "Resting Blood Pressure (trestbps, mm Hg)",
            "chol": "Serum Cholesterol (chol, mg/dl)",
            "age": "Patient Age (years)",
            "oldpeak": "ST Depression (oldpeak)"
        }.get(chosen_numeric, chosen_numeric)

        ax.set_title(f"{display_name} by Target Status", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Diagnosis", fontsize=11)
        ax.set_ylabel(display_name, fontsize=11)
        plt.tight_layout()
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        print(f"[SAVED] Numeric feature comparison chart: {chart3_path.relative_to(project_root)}")
    else:
        print(f"[SKIP] Chart 3 skipped: Neither 'thalach' nor candidate numeric features found.")
        print(f"Available columns: {list(df.columns)}")

    # -------------------------------------------------------------
    # Chart 4: Correlation Heatmap
    # -------------------------------------------------------------
    chart4_path = figures_dir / "correlation_heatmap.png"
    # Exclude target_original to avoid duplicate target representation in correlation
    corr_cols = [c for c in df.columns if c != "target_original"]
    if len(corr_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
            ax=ax
        )
        ax.set_title("Correlation Heatmap of Dataset Attributes", fontsize=14, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(chart4_path, dpi=300)
        plt.close()
        print(f"[SAVED] Correlation heatmap: {chart4_path.relative_to(project_root)}")
    else:
        print(f"[SKIP] Chart 4 skipped: Not enough numeric columns. Found columns: {list(df.columns)}")

    print("\n" + "=" * 65)
    print("EDA Complete. All figures saved successfully to outputs/figures/.")
    print("=" * 65)


if __name__ == "__main__":
    run_eda()
