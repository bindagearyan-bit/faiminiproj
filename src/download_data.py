"""
download_data.py

Fetches the UCI Heart Disease dataset (UCI ID 45) using the ucimlrepo library.
Dataset source: Cleveland Clinic Foundation subset (Detrano et al., 1989).
Saves local CSV at data/raw/heart_disease_cleveland.csv with both:
  - 'target': binary indicator (0 = absence, 1 = presence of heart disease)
  - 'target_original': original angiographic disease score (0 to 4)
Note: Neither target column must ever be used as an input feature during model training.
"""

from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_save_data():
    # Resolve project root relative to this script
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    output_csv = raw_data_dir / "heart_disease_cleveland.csv"

    # Ensure output directory exists
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fetching UCI Heart Disease dataset (ID: 45) via ucimlrepo...")
    print("Subset: Cleveland Database (Robert Detrano et al., 1989)")
    print("=" * 60)

    # Fetch dataset from UCI repository
    heart_disease = fetch_ucirepo(id=45)

    # Features and raw target table
    features = heart_disease.data.features.copy()
    targets = heart_disease.data.targets.copy()

    # The original target column is 'num' (0: <50% stenosis, 1-4: >=50% stenosis)
    target_series = targets["num"] if "num" in targets.columns else targets.iloc[:, 0]

    # Assemble dataset
    df = features.copy()
    # Keep original target (0 to 4) for reference/documentation
    df["target_original"] = target_series.values
    # Create binary target: 0 stays 0; values 1, 2, 3, 4 become 1
    df["target"] = (df["target_original"] > 0).astype(int)

    # Save to local CSV
    df.to_csv(output_csv, index=False)

    # Output verification details
    print(f"\nSuccessfully saved dataset to: {output_csv.relative_to(project_root)}")
    print(f"Dataset Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    
    print("\nColumns found in saved CSV:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nBinary Target counts ('target'):")
    print(df["target"].value_counts().sort_index().to_frame(name="Count").rename(
        index={0: "0 (No Disease)", 1: "1 (Heart Disease)"}
    ))

    print("\nOriginal Target counts ('target_original'):")
    print(df["target_original"].value_counts().sort_index().to_frame(name="Count"))
    print("=" * 60)


if __name__ == "__main__":
    download_and_save_data()
