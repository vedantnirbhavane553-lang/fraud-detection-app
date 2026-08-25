# 🛡️ Real-Time Financial Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-app-vrjec96wntkqxvvgxymrhu.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/vedantnirbhavane553-lang/fraud-detection-app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning web application designed to detect fraudulent financial transactions in real-time. Built with Python, Scikit-Learn, and deployed on Streamlit Cloud.

---

## 🔗 Live Demo
Access the live interactive application here:  
👉 **[Fraud Detection Web App](https://fraud-detection-app-vrjec96wntkqxvvgxymrhu.streamlit.app/)**

---

## 📌 Problem Statement
Financial fraud accounts for billions of dollars in losses annually. The core machine learning challenge in fraud detection is **extreme class imbalance** (fraudulent transactions make up less than 1% of total transactions). This project focuses on building a high-recall, production-ready classifier to flag anomalous transactions while minimizing false positives.

---

## 📊 Dataset & Features
The model was trained on the synthetic financial dataset simulating mobile money transactions:
* **Transaction Types:** `TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `CASH_IN`
* **Key Features:**
  * `amount`: Transaction amount
  * `oldbalanceOrg` / `newbalanceOrig`: Sender's balance change
  * `oldbalanceDest` / `newbalanceDest`: Receiver's balance change
  * **Engineered Features:** Error in balance calculation (`orig_balance_error`, `dest_balance_error`)

---

## ⚙️ Project Architecture & Pipeline
1. **Data Preprocessing & Cleaning:** Handled missing values, outliers, and encoded categorical variables.
2. **Handling Class Imbalance:** Applied technique (e.g., Class Weighting / SMOTE) to optimize minority class detection.
3. **Model Selection & Tuning:** Trained and benchmarked multiple models:
   * Logistic Regression (Baseline)
   * Random Forest Classifier
   * XGBoost / Gradient Boosting
4. **Evaluation Focus:** Prioritized **Recall**, **Precision**, **PR-AUC**, and **ROC-AUC** over raw Accuracy.
5. **Deployment:** Packaged using `joblib` and deployed via **Streamlit Cloud**.

---

## 📈 Model Performance Benchmark

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| **Random Forest / XGBoost** | **0.XX** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

> *Note: Metrics are evaluated on an unseen, stratified test set.*

---

## 🚀 Key App Features
* **Real-Time Scoring:** Enter transaction details to instantly get fraud probability and a binary classification.
* **Dynamic Risk Assessment:** Classifies transactions into Risk Levels (Low, Medium, High).
* **Interactive UI:** Clean Streamlit dashboard with intuitive visual indicators.

---

## 🛠️ Installation & Local Setup

```bash
# 1. Clone the repository
git clone [https://github.com/vedantnirbhavane553-lang/fraud-detection-app.git](https://github.com/vedantnirbhavane553-lang/fraud-detection-app.git)
cd fraud-detection-app

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
