# CardioAI — Heart Disease Risk Screening System

An end-to-end Machine Learning web application designed to assess Coronary Artery Disease (CAD) risk in real time using clinical vitals and diagnostic test readings from the renowned UCI Cleveland Heart Disease dataset.

Built as an academic laboratory project (Department of AI & ML) by **Aryan Bindage**, this system pairs a production-grade Scikit-Learn classification pipeline with an executive clinical dashboard and authentic Jupyter/Colab-style dark visualizations.

> 🩺 **Clinical & Academic Disclaimer:**  
> This software is strictly an educational and laboratory demonstration. It is **NOT** a certified medical diagnostic device or clinical decision support system. Do not use its predictions for patient triage, prescriptions, or clinical decision-making. Always consult certified healthcare practitioners.

---

## 🌟 Key Features

- **⚡ Instant Real-Time Prediction**: Debounced auto-inference delivers model predictions with calibrated probabilities in under 20 milliseconds as you adjust any parameter.
- **📋 13 Validated Clinical Attributes**: Complete intake covering patient demographics, blood chemistry, resting ECG, exercise stress tests, fluoroscopy vessels, and thallium scans.
- **🎨 World-Class Executive Dark UI**: Built with pure Vanilla CSS tokens, radiant verdict auras (Ruby Red for disease detection, Mint Emerald for healthy baselines, Amber for borderline cases), tactile sliders with dynamic progress fills, and WCAG AA contrast.
- **📊 Authentic Python Data Visualizations**: Generates real-time Seaborn and Matplotlib figures (Biomarker Heatmap, Feature Contribution Weights, Cleveland Dataset Scatter, and Decision Confidence Gauge) styled with a seamless dark notebook aesthetic.
- **🧪 One-Click Patient Presets**: Instant benchmark profiles to demonstrate High-Risk, Low-Risk, and Borderline scenarios, plus an arbitrary random profile generator.
- **🔒 Zero Data Leakage**: Standardized preprocessing with median imputation, robust scaling, and one-hot encoding encapsulated within Scikit-Learn `Pipeline` and `ColumnTransformer`.

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
- **Dynamic Single Visualization**: The visualizer automatically selects **only 1** optimal representation from Matplotlib, Seaborn, and Scikit-Learn that best portrays the patient's specific outcome.

---

### Step 6: Run Simultaneously on Google Colab (Cloud Backend)

You can run the model and inference engine directly inside **Google Colab**, while using the website to test inputs:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bindagearyan-bit/faiminiproj/blob/main/notebooks/CardioPulse_Colab_Live_Backend.ipynb)

1. Open [`notebooks/CardioPulse_Colab_Live_Backend.ipynb`](notebooks/CardioPulse_Colab_Live_Backend.ipynb) in Google Colab.
2. Run the notebook cells to launch `!python colab_tunnel.py`.
3. Copy the public tunnel URL printed in Colab (e.g. `https://xxxx.trycloudflare.com` or `https://xxxx.ngrok-free.app`).
4. On the website, click **"☁️ Connect Google Colab"** in the top navigation bar, paste the URL, and click **"Connect"**.
5. As you adjust sliders on the website, Google Colab simultaneously receives the patient vitals, calculates the probability, generates the Seaborn heatmap, and streams the output directly to the website!

---

## Generated File Structure

```text
heart-disease-classifier/
├── app.py                              # Flask server for live clinical web dashboard
├── colab_tunnel.py                     # Google Colab live tunnel and cloud backend runner
├── notebooks/
│   └── CardioPulse_Colab_Live_Backend.ipynb # One-click Colab notebook for faculty viva
├── templates/
│   └── index.html                      # Dark-mode responsive clinical UI & Colab modal
├── static/
│   ├── css/style.css                   # World-class clinical design system & dark theme
│   └── js/app.js                       # Real-time inference, slider sync & Colab cloud bridge
├── requirements.txt                    # Project dependencies (including Flask & pyngrok)

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
