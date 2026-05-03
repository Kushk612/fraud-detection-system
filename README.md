# 🚨 Fraud Detection System (ML + FastAPI)

An end-to-end machine learning system that detects fraudulent transactions using classification models and serves predictions via a FastAPI backend.

---

# 🧠 Problem Statement

Fraudulent transactions are:

- Rare but high impact
- Highly imbalanced
- Difficult to detect in real time

This project builds a system to:

- Detect fraud transactions
- Handle class imbalance
- Provide real-time predictions via API

---

# ⚙️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Joblib

---

# 📊 Dataset

- Kaggle Credit Card Fraud Dataset (2023 version)
- Highly imbalanced dataset (~0.17% fraud cases)
- PCA transformed features (V1–V28)

---

# 🏗️ Project Architecture

Data → Preprocessing → Model Training → Model Selection → API → Prediction

---

# 🚀 Step-by-Step Implementation

## 🟢 Step 1: Dataset Loading

- Dataset fetched using `kagglehub`
- Automatically stored in `/data`
- Loaded using reusable loader script

---

## 🟡 Step 2: Data Preprocessing

- Train-test split with stratification
- Feature scaling using StandardScaler
- Ensured no data leakage (fit only on training data)

---

## 🟠 Step 3: Model Training

Models used:

- Logistic Regression
- Random Forest
- XGBoost

Key focus:

- Class imbalance handling using class weights
- ROC-AUC used as evaluation metric

---

## 🔵 Step 4: Model Evaluation

Evaluated using:

- Precision
- Recall (important for fraud detection)
- F1-score
- ROC-AUC

Best model selected automatically.

---

## 💾 Step 5: Model Saving

- Best model saved using `joblib`
- Stored in `/models/fraud_model.pkl`
- Sample input saved for testing

---

## 🌐 Step 6: FastAPI Integration

Endpoints:

### Health Check

GET /health

### Prediction

POST /predict

### Sample Testing

POST /predict-from-sample

---

## 🧪 Sample Input

```json
{
  "sample": [ ... 30 PCA values ... ]
}
```

---

## 🔮 Output Format

```json
{
  "fraud_prediction": 0,
  "fraud_probability": 0.12,
  "risk": "LOW"
}
```

---

# 📈 Key Features

- End-to-end ML pipeline
- Handles extreme class imbalance
- Multi-model training system
- Automated best model selection
- Real-time API inference
- Structured input validation

---

# 🧠 Key Learnings

- Importance of recall in fraud detection
- Data leakage prevention in ML pipelines
- Model selection based on ROC-AUC
- API design for ML systems
- Production-style ML architecture

---

# 🚀 How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Train model

```bash
python ml/train.py
```

## 3. Start API

```bash
uvicorn app.main:app --reload
```

---

# 📌 Future Improvements

- Add SHAP explainability
- Deploy on cloud (AWS / Render)
- Add Streamlit dashboard
- Add logging & monitoring
- Add Docker support

---
## docs
<img width="1850" height="1053" alt="image" src="https://github.com/user-attachments/assets/abc2f590-aa98-4434-bd3f-a7b81b61b52c" />

## sample test
<img width="1850" height="1053" alt="image" src="https://github.com/user-attachments/assets/3e818ed1-30bd-4432-97da-0a41c8ff6a88" />
<img width="1850" height="1053" alt="image" src="https://github.com/user-attachments/assets/5b0b9044-dc07-46b8-b3c4-12230b2456a6" />
