# ❤️ Heart Failure Prediction using LightGBM

This project predicts whether a heart failure patient is likely to die, using **LightGBM**, advanced preprocessing, and a simple **Gradio interface**.

It uses SMOTE to balance the data, custom feature engineering, and robust scaling for best accuracy.

---

## 📌 Objective

To help doctors and researchers **predict patient survival** using clinical features.  
The app takes patient data as input and predicts if the patient is at risk.

---

## 📁 Dataset

- **Source**: `heart_failure_clinical_records_dataset.csv`
- **Target**: `DEATH_EVENT` (1 = death, 0 = survived)
- **Samples**: 299
- **Original Features**: 12
**Features**:
  - age
  - anaemia
  - creatinine_phosphokinase
  - diabetes
  - ejection_fraction
  - high_blood_pressure
  - platelets
  - serum_creatinine
  - serum_sodium
  - sex
  - smoking
  - time
- **New Engineered Features**:
  - `platelets_per_age`
  - `creatinine_per_ck`
  - `ejection_per_age`
  - `anaemia_creatinine`

---

## 🧠 Model Information

| Component            | Description                              |
|----------------------|-----------------------------------------|
| Model                | `LightGBMClassifier`                     |
| Scaling              | `RobustScaler`                           |
| Balancing            | `SMOTE`                                  |
| Evaluation Method    | Train/Test Split + Stratified CV       |
| Accuracy             | ✅ **98%** (Test set)                   |
| CV Accuracy          | ✅ **~84%** Stratified cross-validation |

---

## 📦 File Structure

```bash
📁 heart-failure-prediction/
│
├── app.py                    # Gradio interface
├── model.ipynb               # Data preprocessing + model training
├── lightgbm_model.pkl        # Saved LightGBM model
├── scaler.pkl                # RobustScaler used on features
├── heart_failure_clinical_records_dataset.csv
├── requirements.txt
├── README.md
```
## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

