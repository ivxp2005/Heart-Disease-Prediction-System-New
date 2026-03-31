# ❤️ Heart Disease Risk Predictor

An interactive web application that predicts a patient's 10-year risk of coronary heart disease (CHD) using machine learning trained on the **Framingham Heart Study** dataset.

---

## Table of Contents

- [❤️ Heart Disease Risk Predictor](#️-heart-disease-risk-predictor)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Live Demo \& Screenshots](#live-demo--screenshots)
  - [Dataset](#dataset)
    - [Input Features (original)](#input-features-original)
  - [How It Works](#how-it-works)
    - [Prediction Pipeline (app)](#prediction-pipeline-app)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection \& Training](#model-selection--training)
    - [Why Logistic Regression?](#why-logistic-regression)
    - [The Class Imbalance Problem](#the-class-imbalance-problem)
    - [Hyperparameters](#hyperparameters)
    - [Preprocessing](#preprocessing)
  - [Performance Metrics](#performance-metrics)
    - [Confusion Matrix (test set, n=848)](#confusion-matrix-test-set-n848)
  - [Prediction Modes](#prediction-modes)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Running the App](#running-the-app)
  - [Retraining the Model](#retraining-the-model)
  - [Sample Test Values](#sample-test-values)
    - [Not At Risk (~4%)](#not-at-risk-4)
    - [Low Risk (~12%)](#low-risk-12)
    - [Moderate Risk (~35%)](#moderate-risk-35)
    - [High Risk (~65%+)](#high-risk-65)
  - [Tech Stack](#tech-stack)
  - [Limitations \& Future Work](#limitations--future-work)
    - [Current Limitations](#current-limitations)
    - [Potential Improvements](#potential-improvements)
  - [Disclaimer](#disclaimer)
  - [👨‍💻 Meet the Team](#-meet-the-team)

---

## Overview

This tool gives healthcare professionals and individuals an instant, explainable risk estimate for 10-year coronary heart disease. Enter patient demographics and clinical measurements in the sidebar — the app returns:

- A **probability percentage** (e.g. 4.3%)
- A **risk category** (Not At Risk / At Risk)
- A **gauge chart** and **feature importance bar chart**
- A plain-English interpretation of what the result means

Three selectable **Prediction Modes** let the user tune the sensitivity vs. precision trade-off without retraining the model.

---

## Live Demo & Screenshots

Run locally — see [Running the App](#running-the-app).

---

## Dataset

| Property | Value |
|---|---|
| Name | Framingham Heart Study |
| File | `framingham.csv` |
| Rows | 4,240 patients |
| Target | `TenYearCHD` (10-year coronary heart disease risk) |
| Positive rate | **15.2%** (severely imbalanced — only 1 in ~7 patients develops CHD) |
| Source | Kaggle / UCI ML Repository |

### Input Features (original)

| Feature | Type | Description |
|---|---|---|
| `male` | Binary | Sex (1 = Male, 0 = Female) |
| `age` | Integer | Age in years |
| `education` | Ordinal | 1 = Some HS → 4 = College Degree |
| `currentSmoker` | Binary | Currently smoking (1 = Yes) |
| `cigsPerDay` | Integer | Cigarettes smoked per day |
| `BPMeds` | Binary | On blood pressure medication |
| `prevalentStroke` | Binary | History of stroke |
| `prevalentHyp` | Binary | Has hypertension |
| `diabetes` | Binary | Has diabetes |
| `totChol` | Float | Total cholesterol (mg/dL) |
| `sysBP` | Float | Systolic blood pressure (mmHg) |
| `diaBP` | Float | Diastolic blood pressure (mmHg) |
| `BMI` | Float | Body mass index |
| `heartRate` | Integer | Resting heart rate (bpm) |
| `glucose` | Float | Fasting glucose (mg/dL) |

---

## How It Works

```
User Input (sidebar)
     ↓
Feature Engineering  →  adds 5 interaction features (20 total)
     ↓
Median Imputation    →  handles missing values
     ↓
StandardScaler       →  normalises all features to mean=0, std=1
     ↓
Logistic Regression  →  outputs CHD probability (0–100%)
     ↓
Threshold Comparison →  probability ≥ threshold? → At Risk : Not At Risk
     ↓
Display Results      →  gauge, bar chart, plain-English explanation
```

### Prediction Pipeline (app)

1. Collect 15 raw inputs from sidebar.
2. Compute 5 engineered features.
3. Build a 20-feature DataFrame in the exact order the model was trained on.
4. Apply `bundle['imputer'].transform()` then `bundle['scaler'].transform()`.
5. Call `model.predict_proba()` → pick column 1 (CHD probability).
6. Compare against `selected_mode['threshold']` → binary prediction.
7. Render results.

---

## Feature Engineering

Five interaction terms are added to capture non-linear relationships:

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `pulse_pressure` | `sysBP − diaBP` | Arterial stiffness indicator |
| `age_sysBP` | `age × sysBP` | Compound cardiovascular risk with age |
| `smoke_age` | `currentSmoker × age` | Cumulative smoking damage over lifetime |
| `glucose_diabetes` | `glucose × (diabetes + 1)` | Amplifies glucose risk when diabetic |
| `smoking_burden` | `currentSmoker × cigsPerDay` | Total daily smoking load |

---

## Model Selection & Training

### Why Logistic Regression?

Several models were evaluated. On this 4,240-row dataset with 15% positive rate, Logistic Regression consistently outperformed more complex models:

| Model | ROC-AUC | Notes |
|---|---|---|
| **Logistic Regression** | **70.16%** | Best — simple model fits small dataset well |
| Random Forest | 67.4% | Overfits on small data |
| XGBoost | 63.9% | Further overfits; needs more data |
| Stacking Ensemble | 67.2% | Complexity not rewarded here |

### The Class Imbalance Problem

With 85% negatives, a naive model trained at threshold=0.5 almost **never** predicts CHD — achieving high accuracy (67%) by doing nothing useful. To fix this:

1. **`class_weight='balanced'`** — penalises misclassifying the minority class more heavily during training.
2. **Youden's J threshold tuning** — instead of defaulting to 0.5, find the threshold `t` that maximises `sensitivity + specificity − 1`:

$$J = \max_t \left( \text{TPR}(t) - \text{FPR}(t) \right)$$

This shifts `t` from `0.5 → 0.3364`, dramatically improving recall for CHD cases.

### Hyperparameters

```python
LogisticRegression(
    C=0.5,                    # moderate L2 regularisation
    class_weight='balanced',  # compensates 85/15 imbalance
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
```

### Preprocessing

```
SimpleImputer(strategy='median')  →  fills NaN with column median
StandardScaler()                  →  zero mean, unit variance
```

---

## Performance Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | 70.16% | Model ranks a random CHD patient above a random healthy patient 70% of the time |
| **Accuracy** | 50.12% | Low because the tuned threshold accepts more false alarms to catch more real cases |
| **F1-Score** | 0.3522 | Harmonic mean of precision and recall for the CHD class |
| **Sensitivity (Recall)** | 89.15% | Catches 89 out of every 100 CHD patients |
| **Specificity** | 43.12% | Correctly clears 43% of healthy patients |
| **False Alarm Rate** | ~57% | Of those flagged "At Risk", ~57% are actually healthy |

### Confusion Matrix (test set, n=848)

```
                  Predicted Healthy   Predicted CHD
Actual Healthy         310                409
Actual CHD              14                115
```

**Why accuracy is 50% but ROC-AUC is 70%** — Accuracy counts all errors equally. With the tuned threshold we deliberately accept more false alarms (409) to avoid missing real cases (only 14 missed). ROC-AUC measures the model's overall *ranking* ability independent of any threshold, and 70% is a genuinely informative model.

---

## Prediction Modes

Three modes are available via the **⚙️ Prediction Mode** selector in the sidebar. They use the same trained model — only the decision threshold changes:

| Mode | Threshold | Sensitivity | False Alarm Rate | Use Case |
|---|---|---|---|---|
| 🔴 High Sensitivity | 0.336 | 89% | ~57% | Screening — catch every possible case |
| 🟡 Balanced (default) | 0.400 | 76% | ~47% | General use |
| 🟢 High Precision | 0.450 | 67% | ~40% | Confirmatory — reduce unnecessary follow-up |

**Switching modes changes the risk label** (At Risk / Not At Risk) shown in the app. The underlying probability does not change.

---

## Project Structure

```
heart disease predictor/
├── framingham.csv              # Framingham Heart Study dataset
├── heart_disease_app.py        # Streamlit web application (main entry point)
├── train_model.py              # Model training script — run to retrain
├── heart_disease_model.pkl     # Saved model bundle (auto-generated)
├── model_info.pkl              # Performance metadata (auto-generated)
├── feature_names.pkl           # Feature list (auto-generated)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# 1. Clone / download the project folder
cd "heart disease predictor"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
# Make sure the virtual environment is active, then:
python -m streamlit run heart_disease_app.py
```

Open **http://localhost:8501** in your browser.

> The app loads with a pre-filled **no-risk profile** (26-year-old healthy female) so the first prediction shown is "Not At Risk".

---

## Retraining the Model

Run this any time you want to retrain from scratch on `framingham.csv`:

```bash
python train_model.py
```

This will:
1. Re-engineer features
2. Retrain Logistic Regression with `class_weight='balanced'`
3. Re-tune the Youden's J threshold
4. Overwrite `heart_disease_model.pkl` and `model_info.pkl`
5. Print a full evaluation report to the console

Then restart the Streamlit app to load the new model.

---

## Sample Test Values

Use these in the sidebar to verify the app is responding correctly:

### Not At Risk (~4%)
| Field | Value |
|---|---|
| Gender | Female |
| Age | 26 |
| Education | College Degree |
| Smoker | No |
| BP Meds / Stroke / Hypertension / Diabetes | No |
| Total Cholesterol | 170 |
| Systolic BP | 108 |
| Diastolic BP | 68 |
| BMI | 21.5 |
| Heart Rate | 62 |
| Glucose | 74 |

### Low Risk (~12%)
| Field | Value |
|---|---|
| Gender | Male |
| Age | 35 |
| Smoker | No |
| Total Cholesterol | 185 |
| Systolic BP | 115 |
| Diastolic BP | 75 |
| BMI | 24.0 |
| Heart Rate | 68 |
| Glucose | 85 |

### Moderate Risk (~35%)
| Field | Value |
|---|---|
| Gender | Male |
| Age | 52 |
| Smoker | Yes, 15 cigs/day |
| Hypertension | Yes |
| Total Cholesterol | 240 |
| Systolic BP | 148 |
| Diastolic BP | 92 |
| BMI | 28.5 |
| Heart Rate | 80 |
| Glucose | 105 |

### High Risk (~65%+)
| Field | Value |
|---|---|
| Gender | Male |
| Age | 65 |
| Smoker | Yes, 30 cigs/day |
| BP Meds | Yes |
| Hypertension | Yes |
| Diabetes | Yes |
| Total Cholesterol | 290 |
| Systolic BP | 185 |
| Diastolic BP | 110 |
| BMI | 33.0 |
| Heart Rate | 92 |
| Glucose | 210 |

---

## Tech Stack

| Component | Library / Version |
|---|---|
| Web framework | Streamlit 1.54.0 |
| ML model | scikit-learn 1.8.0 — LogisticRegression |
| Preprocessing | scikit-learn — SimpleImputer, StandardScaler |
| Model persistence | joblib 1.5.3 |
| Data manipulation | pandas 2.3.3, numpy 2.4.2 |
| Charts | plotly 6.5.2 |
| Language | Python 3.8+ |

---

## Limitations & Future Work

### Current Limitations

- **Dataset size**: 4,240 patients is small for a clinical ML model. Larger datasets would benefit from more complex models (XGBoost, neural networks).
- **False alarm rate**: ~57% at High Sensitivity mode — roughly half of people flagged "At Risk" are actually healthy. This is inherent to the dataset's low positive rate.
- **No temporal data**: `cigsPerDay` and other features are snapshots, not longitudinal.
- **Population bias**: Framingham data is predominantly white/American — predictions may be less accurate for other populations.

### Potential Improvements

- [ ] Incorporate more features (LDL/HDL cholesterol, family history, physical activity)
- [ ] Train on a larger, more diverse dataset
- [ ] Add SHAP values for per-patient feature attribution
- [ ] Deploy as a cloud-hosted app (Streamlit Cloud, Azure App Service)
- [ ] Add patient history tracking across sessions

---

## Disclaimer

> This tool is intended for **educational and research purposes only**. It is not a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

<div align="center">

## 👨‍💻 Meet the Team

*Built with passion, code, and a lot of coffee ☕*

<br>


 | **Nazreen Shemeem**  | 
 | **Pardhiv Suresh M** | 
 | **Mohamed Ibrahim**  | 
 | **Zehbia Zulfikar**  | 

<br>

---

*"Predicting tomorrow's health risks, today."*

<br>

![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ML Model](https://img.shields.io/badge/Model-Logistic%20Regression-4CAF50?style=for-the-badge&logo=scikit-learn&logoColor=white)

<br>

© 2026 Heart Disease Risk Predictor Team · All Rights Reserved

</div>

