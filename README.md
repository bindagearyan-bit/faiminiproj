# Heart Disease Classifier (UCI Cleveland Dataset)

An educational, beginner-friendly Python machine learning project demonstrating end-to-end binary classification on clinical cardiac data, built with strict adherence to ML best practices and data leakage prevention.

> ⚠️ **IMPORTANT MEDICAL DISCLAIMER: FOR EDUCATIONAL USE ONLY**  
> This project, code, and trained machine learning pipelines are developed solely for educational and academic purposes. They are **NOT** medical devices, diagnostic instruments, or clinical decision support tools. **DO NOT USE THIS PROJECT OR ITS PREDICTIONS FOR ANY MEDICAL DIAGNOSIS, TREATMENT, OR CLINICAL DECISION-MAKING.** Always seek the advice of a qualified physician or healthcare provider with any medical questions.

---

## Project Goal

The goal of this project is to provide a clean, complete, and reproducible machine learning pipeline that predicts the presence ($1$) or absence ($0$) of coronary artery disease from clinical patient features while helping learners understand how machine learning models learn from data.

Key pedagogical and engineering principles implemented:
- **Zero Target Leakage**: Isolates predictor attributes from both binary and original multi-class targets.
- **Leak-Free Pipelines**: Encapsulates imputation, scaling, and one-hot encoding inside scikit-learn `Pipeline` and `ColumnTransformer`.
- **Honest Model Selection**: Compares a `DummyClassifier` baseline, `LogisticRegression`, and `RandomForestClassifier` using 5-fold stratified cross-validation on the **training set only**.
- **Unbiased Test Evaluation**: Evaluates the winning pipeline only **once** on an untouched, held-out test set ($25\%$).

---

## Dataset Source and Citation

- **Dataset**: Heart Disease (Cleveland Database)
- **UCI Repository ID**: [45](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Fetching Library**: `ucimlrepo` Python package
- **Donors/Creators**:
  - Robert Detrano, M.D., Ph.D. (Cleveland Clinic Foundation)
  - Andras Janosi, M.D. (Hungarian Institute of Cardiology, Budapest)
  - William Steinbrunn, M.D. (University Hospital, Zurich, Switzerland)
  - Matthias Pfisterer, M.D. (University Hospital, Basel, Switzerland)
- **Primary Scientific Citation**:
  > Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. **The American Journal of Cardiology**, 64(5), 304–310. [https://doi.org/10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9)
- **UCI Repository Citation**:
  > Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease*. UCI Machine Learning Repository. [https://doi.org/10.24432/C52P4X](https://doi.org/10.24432/C52P4X)

---

## Setup Commands

### 1. (Optional) Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies
```bash
pip install -r requirements.txt
```

---

## Execution Guide

Run the three scripts in the exact sequence shown below. Each script uses project-root-relative paths and can be executed from any terminal directory.

### Step 1: Download & Prepare Dataset
```bash
python src/download_data.py
```
- Fetches the Cleveland dataset (UCI ID 45) using `ucimlrepo`.
- Preserves `target_original` ($0$ to $4$) and constructs binary `target` ($0 = \text{no disease}, 1 = \text{disease}$).
- Saves local CSV to `data/raw/heart_disease_cleveland.csv`.

### Step 2: Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
- Inspects dataset dimensions, missing values (`ca`: 4, `thal`: 2), and distribution statistics.
- Generates and saves 4 publication-quality charts in `outputs/figures/`.

### Step 3: Model Training & Evaluation
```bash
python src/train_model.py
```
- Performs a stratified $75/25$ train/test split.
- Compares models using 5-fold Stratified CV on the training set.
- Selects the winning pipeline, trains it on all training data, and evaluates it once on the test set.
- Exports confusion matrix graphic, metrics CSVs, and the fitted pipeline artifact.

### Step 4: Classify Any Custom Values (Live Inference)
```bash
# Option A: Interactive step-by-step terminal prompt (guides you through all 13 features)
python src/predict.py

# Option B: Pass custom patient features directly via CLI flags
python src/predict.py --age 58 --sex 1 --cp 4 --trestbps 140 --chol 260 --fbs 0 --restecg 1 --thalach 130 --exang 1 --oldpeak 2.2 --slope 2 --ca 1 --thal 7

# Option C: Run instant demonstration on preconfigured profiles (High, Low, and Moderate Risk)
python src/predict.py --demo

# Option D: Classify via JSON file or JSON string
python src/predict.py --json '{"age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 210, "thalach": 165, "oldpeak": 0.4, "slope": 1, "ca": 0, "thal": 3}'

# Option E: Batch classify a full CSV file
python src/predict.py --csv path/to/patients.csv
```
- Accepts **any arbitrary patient values** provided by the user (not limited to training or historical data).
- Applies identical preprocessing automatically (median imputation, standardization, one-hot encoding).
- Outputs the classification:
  * **Class 0 (Negative / No Heart Disease)** or **Class 1 (Positive / Heart Disease Detected)**
  * **Precise Probabilities & Confidence** (e.g. `98.71%` Disease, `1.29%` Absence)
  * **Assessed Risk Tier** (Low Risk, Moderate Risk, High Risk)
  * **Key Clinical Risk Factor Breakdown** (e.g. ST depression, asymptomatic angina, fluoroscopy vessels, thallium defect)

---

## Verified Experimental Results

These results were produced by running the full pipeline on the dataset:

### 1. Training Set 5-Fold Stratified Cross-Validation (Model Selection)
| Model | Mean CV F1 | Std CV F1 | Winner? |
| :--- | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | $0.0000$ | $0.0000$ | Baseline |
| **Logistic Regression** | **$0.7971$** | **$0.0441$** | **Yes (Chosen)** |
| **Random Forest Classifier** | $0.7911$ | $0.0628$ | No |

### 2. Held-Out Test Set Evaluation ($N=76$ Patients)
The winning **Logistic Regression Pipeline** achieved the following on the untouched test set:
- **Accuracy**: $85.53\%$ ($65 / 76$ correct)
- **Precision**: $83.33\%$ ($30 / 36$ positive predictions correct)
- **Recall**: $85.71\%$ ($30 / 35$ actual cases detected)
- **F1 Score**: $84.51\%$

**Test Confusion Matrix:**
- **True Negatives (TN)**: $35$ (Healthy correctly predicted healthy)
- **False Positives (FP)**: $6$ (Healthy incorrectly predicted disease)
- **False Negatives (FN)**: $5$ (Sick patients missed)
- **True Positives (TP)**: $30$ (Sick patients correctly identified)

---

### Step 5: Launch Live Interactive Web Diagnostic Dashboard
```bash
python app.py
```
- Starts the live clinical diagnostic web application at:
  - **Local Web URL**: [http://127.0.0.1:5000](http://127.0.0.1:5000) or [http://localhost:5000](http://localhost:5000)
- **Arbitrary Real-Time Input**: Faculty can test ANY custom patient attributes via interactive sliders and segmented controls without relying on pre-recorded dataset records.
- **Dynamic Single Visualization**: The visualizer automatically selects **only 1** optimal representation from Matplotlib, Seaborn, and Scikit-Learn (Clinical Biomarker Heatmap, Model Log-Odds Attribution, ECG ST-Segment Waveform, or Cardiac Stress Space) that best portrays the patient's specific outcome.

---

## Generated File Structure

```text
heart-disease-classifier/
├── app.py                              # Flask server for live clinical web dashboard
├── templates/
│   └── index.html                      # Dark-mode responsive clinical UI
├── static/
│   ├── css/style.css                   # Medical dashboard styling & glassmorphism
│   └── js/app.js                       # Real-time inference & slider synchronization
├── requirements.txt                    # Project dependencies (including Flask)
├── README.md                           # Project overview, citation, commands, results
├── learning_walkthrough.md             # In-depth beginner ML concepts & tutorial
├── data/
│   └── raw/
│       └── heart_disease_cleveland.csv # Processed dataset with target and target_original
├── models/
│   └── heart_disease_pipeline.joblib   # Trained and serialized winning pipeline
├── outputs/
│   ├── figures/
│   │   ├── age_distribution.png        # Patient age histogram with mean/median
│   │   ├── confusion_matrix.png        # Test set confusion matrix visualization
│   │   ├── correlation_heatmap.png     # Feature correlation matrix
│   │   ├── numeric_feature_by_target.png # Max heart rate (thalach) by diagnosis
│   │   └── target_counts.png           # Target class balance bar chart
│   └── metrics/
│       ├── cv_model_comparison.csv     # 5-fold CV F1 scores on training set
│       └── model_evaluation_metrics.csv # Final test set evaluation metrics
└── src/
    ├── download_data.py                # Dataset acquisition and preprocessing
    ├── eda.py                          # Exploratory data analysis and charting
    ├── train_model.py                  # Training, cross-validation, and evaluation
    ├── predict.py                      # Interactive & CLI inference on custom patient values
    └── visualization.py                # Dynamic single-best Matplotlib/Seaborn visualizer
```

---

## Pedagogical Guide

For a detailed explanation of what $X$, $y$, features, targets, `fit()`, `predict()`, precision, recall, and confusion matrices mean in plain English, read [learning_walkthrough.md](learning_walkthrough.md).
