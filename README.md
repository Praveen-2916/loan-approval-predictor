# 🏦 Loan Approval Prediction Web App

A Machine Learning web application built with **Python**, **scikit-learn**, and **Streamlit** that predicts whether a bank loan application will be approved or rejected based on user financial metrics.

---

## 🚀 Live Demo
👉 **[Click Here to Try the Live App](https://loan-approval-predictor-sj3jksmqotd69smqbdxhvz.streamlit.app/)**

---

## 📌 Features
* **Interactive UI:** Built using Streamlit for fast and clean user input forms.
* **Full Data Preprocessing:** Handles missing values (imputation), categorical encoding (One-Hot & Ordinal), feature scaling, and feature transformation.
* **Predictive Model:** Powered by a Logistic Regression classifier trained on applicant financial data.

---

## 🛠️ Machine Learning Workflow & Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`
* **Preprocessing Steps:**
  1. Mean Imputation for missing numerical values
  2. Mode Imputation for missing categorical values
  3. Ordinal Encoding for `Education_Level`
  4. One-Hot Encoding for remaining categorical variables
  5. Feature Squaring applied to `Credit_Score` and `DTI_Ratio`
  6. Standard Scaling applied across all features
* **Model:** Logistic Regression

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Praveen-2916/loan-approval-predictor.git](https://github.com/Praveen-2916/loan-approval-predictor.git)
   cd loan-approval-predictor
