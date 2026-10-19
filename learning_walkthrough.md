# Understanding How Machine Learning Learns: A Beginner's Walkthrough

Welcome to the **Heart Disease Classifier** learning walkthrough! This guide breaks down the core concepts of machine learning (ML) using plain English, intuitive medical analogies, and concrete examples from our project.

---

## 1. The Core Concept: Features ($X$) and Target ($y$)

In traditional programming, a software developer writes explicit rules:
$$\text{Input Data} + \text{Handwritten Rules} \longrightarrow \text{Output}$$

In **Machine Learning**, the computer discovers the rules automatically by examining examples:
$$\text{Input Data } (X) + \text{Known Outcomes } (y) \longrightarrow \text{Learned Model / Rules}$$

### What is a Feature?
A **feature** is an individual measurable property or clinical observation of a patient. In our dataset, each row represents one patient, and each feature column describes an attribute of that patient:
- `age`: Age in years
- `sex`: Biological sex ($1 = \text{male}, 0 = \text{female}$)
- `cp`: Chest pain type ($1 = \text{typical angina}, 2 = \text{atypical angina}, 3 = \text{non-anginal}, 4 = \text{asymptomatic}$)
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol (in mg/dl)
- `thalach`: Maximum heart rate achieved during exercise stress test
- `exang`: Exercise-induced angina ($1 = \text{yes}, 0 = \text{no}$)
- `oldpeak`: ST depression induced by exercise relative to rest

### Why is $X$ Capitalized?
In mathematics and Python convention:
- **$X$** is capitalized because it is a **2D matrix** (a table with multiple rows and multiple columns). For our training set, $X$ has shape `(227, 13)`: 227 patient rows and 13 clinical feature columns.
- **$y$** is lowercase because it is a **1D vector** (a single column of answers). For our training set, $y$ has length `227`.

### What is the Target ($y$)?
The **target** is the specific answer or outcome we want the model to learn to predict:
- `0`: Absence of heart disease ($<50\%$ diameter narrowing in coronary arteries).
- `1`: Presence of heart disease ($\ge 50\%$ diameter narrowing).

### Target Leakage: The Golden Rule
In our data preparation step, we stored the original 5-level clinical score in `target_original` and our binary label in `target`.
> **Never include `target` or `target_original` inside $X$.**  
> If an algorithm had access to the answer column during training, it would just memorize `if target == 1: predict 1`. It would score 100% on training, but be completely useless on real patients who don't already have a known diagnosis.

---

## 2. Train / Test Split: The Generalization Test

Imagine preparing for a final medical exam:
- If a student practices on 100 sample questions and the final exam contains **the exact same 100 questions**, a high score only proves **memorization**, not **understanding**.
- To test if the student truly understands the underlying concepts, the final exam must contain **new, unseen questions**.

### How We Split
We split our 303 patients into two strictly separate groups:
1. **Training Set ($75\% = 227\text{ patients}$)**: Used by algorithms to discover relationships and patterns.
2. **Held-Out Test Set ($25\% = 76\text{ patients}$)**: Locked away in a "vault". The models are **never** allowed to see this data during training or model selection. It is evaluated only once at the very end to measure real-world performance.

### What is "Stratification"?
If $54\%$ of patients in the whole dataset have no disease and $46\%$ have disease, a random split could accidentally put all sick patients into the test set by chance. **Stratified splitting** guarantees that both the train set and test set maintain the exact same $54\% / 46\%$ ratio.

---

## 3. What `fit()` Means by "Learning"

When you call:
```python
pipeline.fit(X_train, y_train)
```
What is the computer actually doing?

### In Logistic Regression: Finding Mathematical Weights
Logistic regression models the probability $P$ that a patient has heart disease using an equation:
$$z = w_1 \cdot \text{age} + w_2 \cdot \text{thalach} + w_3 \cdot \text{chol} + \dots + b$$
$$P(\text{Heart Disease}) = \frac{1}{1 + e^{-z}}$$

1. **Initialization**: The computer starts with random guesses for all weights ($w_1, w_2, \dots$) and the intercept ($b$).
2. **Error Calculation**: It computes predictions for all training patients and compares them to actual $y_{\text{train}}$. It calculates a loss/penalty whenever it makes an error.
3. **Adjustment (Gradient Descent)**: It systematically nudges each weight up or down to reduce the error. For example, if higher `thalach` (max heart rate) is correlated with lower heart disease risk, the weight for `thalach` becomes negative.
4. **Convergence**: Once additional adjustments no longer decrease the error, learning is complete!

### In Random Forest: Growing an Ensemble of Decision Trees
A Random Forest builds 100 distinct decision trees. Each tree creates a sequence of binary questions:
- *Question 1*: Is `thalach` $< 148$?
  - *If Yes*: Is `cp` (chest pain) $= 4$ (asymptomatic)?
    - *If Yes*: Predict Heart Disease ($1$).
    - *If No*: Predict Healthy ($0$).
- *If No*: Check `oldpeak`...

Each tree is trained on a random sample of patients and a random subset of features. When making a prediction, all 100 trees vote, and the majority vote wins.

---

## 4. What `predict()` Does

When training is done, we pass new, unseen patients to:
```python
y_pred = pipeline.predict(X_test)
```
1. The pipeline automatically imputes missing values and scales numeric columns using the parameters learned during training.
2. The classifier computes the estimated probability of disease $P(\text{Disease} \mid X)$.
3. If $P \ge 0.50$, it outputs **1** (Heart Disease). If $P < 0.50$, it outputs **0** (No Disease).

---

## 5. Evaluation Metrics & The Confusion Matrix

In medical diagnosis, raw accuracy alone can be deceptive. We must examine where errors happen.

### The Confusion Matrix Explained
Our test set has 76 patients. The trained model produced this confusion matrix:

| Actual \ Predicted | Predicted Healthy ($0$) | Predicted Disease ($1$) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Healthy ($0$)** | **$35$ True Negatives (TN)** | **$6$ False Positives (FP)** | 41 patients |
| **Actual Disease ($1$)** | **$5$ False Negatives (FN)** | **$30$ True Positives (TP)** | 35 patients |
| **Total Predicted** | 40 predicted healthy | 36 predicted disease | 76 total |

- **True Negative (TN = 35)**: Correct diagnosis! Healthy patient correctly identified as healthy.
- **False Positive (FP = 6)**: *False alarm!* Healthy patient incorrectly flagged as having disease. In clinical practice, this leads to follow-up testing and stress, but the error is caught upon further examination.
- **False Negative (FN = 5)**: *Dangerous miss!* Sick patient incorrectly told they are healthy. This is the most dangerous medical mistake, as the patient might miss lifesaving intervention.
- **True Positive (TP = 30)**: Correct diagnosis! Sick patient correctly identified and eligible for care.

### Key Metrics Formulated Simply

1. **Accuracy**: Total percentage of correct predictions:
   $$\text{Accuracy} = \frac{\text{Correct}}{\text{Total}} = \frac{35 + 30}{76} = \frac{65}{76} \approx \mathbf{85.53\%}$$

2. **Precision**: When the model flags a patient as having heart disease, how often is it right?
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{30}{30 + 6} = \frac{30}{36} \approx \mathbf{83.33\%}$$
   *High precision means few false alarms.*

3. **Recall (Sensitivity)**: Out of all patients who genuinely have heart disease, what fraction did our model catch?
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{30}{30 + 5} = \frac{30}{35} \approx \mathbf{85.71\%}$$
   *High recall means few missed diseases.*

4. **F1 Score**: The harmonic mean of precision and recall:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.8333 \times 0.8571}{0.8333 + 0.8571} \approx \mathbf{84.51\%}$$
   *The F1 score provides a balanced view that penalizes models that sacrifice recall for precision or vice versa.*

### Why the Dummy Classifier Baseline Matters
A simple baseline model that simply predicts the most common class (`0`) achieves:
- Accuracy: $\sim 54\%$ (simply guessing $0$ for everyone)
- Precision: $0.00$
- Recall: $0.00$
- **F1 Score: $0.00$**  
Because it never predicts class `1`, it catches zero sick patients! Comparing our models against this baseline proves that our algorithms have genuinely learned useful patterns.

---

## 6. Cross-Validation and Data Leakage Prevention

### Why Cross-Validation on the Training Set?
If we tuned models directly on the test set, we would be "snooping" at the exam answers. Instead, we use **5-Fold Stratified Cross-Validation** on the training set:
- Split the 227 training patients into 5 equal subsets (folds).
- Train on 4 folds, test on the 5th fold.
- Repeat 5 times so every fold acts as a temporary validation set once.
- Compute the average F1 score:
  - **Logistic Regression**: Mean CV F1 $= \mathbf{0.7971}$ (Winner)
  - **Random Forest**: Mean CV F1 $= 0.7911$
  - **Dummy Baseline**: Mean CV F1 $= 0.0000$

### Why Use `Pipeline`?
When scaling data (e.g. subtracting mean and dividing by standard deviation) or imputing missing values, those calculations must be derived **only from training data**. If you calculate the average age of all patients before splitting, the test set's mean "leaks" into the training process.  
Scikit-learn `Pipeline` automatically encapsulates all transformations so that statistics are computed exclusively on training folds.

---

## 7. Step-by-Step Command Execution Guide

Run each command in sequence from the project root:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
*Purpose*: Installs required open-source libraries (`ucimlrepo`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`).

---

### Step 2: Fetch and Prepare the Dataset
```bash
python src/download_data.py
```
*Purpose*: Connects to the UCI Machine Learning Repository (dataset ID 45), fetches the Cleveland clinical dataset, isolates features, generates the binary `target` ($0$ or $1$) while retaining `target_original` ($0$ to $4$), and saves the local CSV to `data/raw/heart_disease_cleveland.csv`.

---

### Step 3: Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
*Purpose*: Reads the local CSV, checks for missing data (finding 4 in `ca` and 2 in `thal`), prints summary statistics, and saves 4 publication-quality visualizations into `outputs/figures/`:
1. `target_counts.png` (Class balance)
2. `age_distribution.png` (Patient age histogram and KDE)
3. `numeric_feature_by_target.png` (Max heart rate comparison by diagnosis)
4. `correlation_heatmap.png` (Feature correlation matrix)

---

### Step 4: Train and Evaluate Models
```bash
python src/train_model.py
```
*Purpose*:
1. Separates features ($X$) and target ($y$) with zero leakage.
2. Performs stratified $75/25$ split.
3. Compares Dummy Baseline, Logistic Regression, and Random Forest using 5-fold Stratified CV F1 on the training set.
4. Identifies the winning model (**Logistic Regression**, CV F1 $= 0.7971$).
5. Fits the winning pipeline on all training data.
6. Evaluates once on the held-out test set ($85.53\%$ Accuracy, $84.51\%$ F1).
7. Saves artifacts:
   - `models/heart_disease_pipeline.joblib` (deployable model pipeline)
   - `outputs/figures/confusion_matrix.png` (visual confusion matrix)
   - `outputs/metrics/model_evaluation_metrics.csv` (test metrics table)
   - `outputs/metrics/cv_model_comparison.csv` (CV comparison table)
