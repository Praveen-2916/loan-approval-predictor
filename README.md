# Loan Approval Prediction

A machine learning project that predicts whether a loan application is likely to be approved based on the applicant's financial information.

I built this project to practice classification, data preprocessing, feature engineering, and deploying a machine learning model as a simple Streamlit web app.

## Live Demo

[Try the Loan Approval Predictor](https://loan-approval-predictor-sj3jksmqotd69smqbdxhvz.streamlit.app/)

## What I Did

- Cleaned and prepared the loan application data
- Handled missing numerical and categorical values
- Encoded categorical features
- Applied feature engineering to `Credit_Score` and `DTI_Ratio`
- Scaled the features
- Compared Logistic Regression, Naive Bayes, and KNN
- Selected Logistic Regression based on the model evaluation
- Built a Streamlit interface for making predictions

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Run Locally

```bash
git clone https://github.com/Praveen-2916/loan-approval-predictor.git
cd loan-approval-predictor
pip install -r requirements.txt
streamlit run app.py