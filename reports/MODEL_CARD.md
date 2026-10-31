# Model Card: PartSense Component Failure & Warranty Risk Engine

## Model Overview
- **Name**: PartSense Multi-Stage Failure Prediction & Anomaly Architecture
- **Version**: 0.1.0
- **Generated At**: 2026-10-31T18:15:20.634200+00:00
- **Primary Algorithm**: Gradient Boosted Trees (XGBoost) with Class Imbalance Weighting (`scale_pos_weight`)
- **Baseline Benchmark**: Standardized Logistic Regression
- **Auxiliary Models**: One-vs-Rest XGBoost Failure Mode Diagnosers & Isolation Forest (Normal-Only Unsupervised Anomaly Engine)
- **License**: MIT (Code), CC BY 4.0 (AI4I Telemetry Benchmark)

---

## Intended Use
- **Primary Domain**: Automotive aftersales, warranty forecasting, and CNC industrial milling condition monitoring.
- **Intended Users**: Fleet reliability engineers, aftersales warranty managers, and plant operations technicians.
- **Decision Workflow**:
  1. Real-time telemetry ingestion from in-service components.
  2. Physics feature engineering (power, temperature differential, strain).
  3. Probability calculation and threshold policy evaluation at tuned boundary ($p^* = 0.3927$).
  4. Root-cause multi-mode diagnosis and Isolation Forest anomaly screening.
  5. Explainable SHAP waterfall decomposition of top physical drivers.

---

## Data Provenance & Strict Leakage Prevention
- **Dataset**: UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset, ID 601, Matzka 2020).
- **Synthetic Notice**: Clearly documented synthetic benchmark reflecting real cutting tool degradation physics.
- **Strict Leakage Rules**: Never use identifiers (`UDI`, `Product ID`) or downstream failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) as features for binary failure prediction.

---

## Audited Evaluation Metrics (Holdout Test Split: 2,000 Samples, 68 Failures)

### 1. Primary Binary Classifier Comparison

| Model Architecture | Test ROC-AUC | Test PR-AUC | Tuned Policy F1 | Tuned Policy Recall | Precision |
|---|:---:|:---:|:---:|:---:|:---:|
| **PartSense XGBoost (Champion)** | **0.9825** | **0.8882** | **0.6492** | **91.2%** | **50.4%** |
| Baseline Logistic Regression | 0.9385 | 0.4297 | 0.2993 | 88.2% | 18.0% |

### 2. Failure-Mode Diagnosis (One-vs-Rest)

| Failure Mode | Physical Meaning | Learnability Status | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|:---:|:---:|:---:|:---:|
| **TWF** | Tool Wear Failure | Learnable | 0.8203 | 0.0622 | 0.0962 |
| **HDF** | Heat Dissipation Failure | Learnable | 0.9998 | 0.9868 | 0.9508 |
| **PWF** | Power Surge / Stall Failure | Learnable | 0.9998 | 0.9654 | 0.9286 |
| **OSF** | Mechanical Overstrain Failure | Learnable | 1.0000 | 1.0000 | 0.9697 |
| **RNF** | Random Electronic Failure | **Not Learnable (Random Noise)** | N/A | N/A | N/A |

### 3. Isolation Forest Anomaly Detection (Trained on Normal Data Only)

| Anomaly Inspection Budget | Inspected Components | Detected True Failures | Precision@k | Recall@k |
|---|:---:|:---:|:---:|:---:|
| **Top 1% Most Anomalous** | 20 | 9 | 45.0% | 13.2% |
| **Top 5% Most Anomalous** | 100 | 27 | 27.0% | 39.7% |
| **Top 10% Most Anomalous** | 200 | 39 | 19.5% | 57.4% |
| **Top 20% Most Anomalous** | 400 | 48 | 12.0% | 70.6% |

---

## Limitations & Ethical Considerations
- **Synthetic Degradation Assumptions**: Model is trained on AI4I synthetic dynamics. Real-world physical variations (lubrication degradation, vibration spectra) must be validated prior to safety-critical deployment.
- **Cold Start & Unseen Failure Modes**: Unprecedented failure types (e.g., contamination, severe corrosion) cannot be classified by supervised heads; operators should monitor the Isolation Forest anomaly score for novel degradation patterns.
