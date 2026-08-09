import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the applicant details to predict loan approval status.")


# Load trained artifacts
@st.cache_resource
def load_artifacts():
    return {
        "num_imp": joblib.load("num_imp.pkl"),
        "cat_imp": joblib.load("cat_imp.pkl"),
        "oe": joblib.load("oe.pkl"),
        "ohe": joblib.load("ohe.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "model": joblib.load("log_model.pkl"),
        "columns": joblib.load("feature_columns.pkl"),
    }


artifacts = load_artifacts()

with st.form("loan_form"):
    st.subheader("1. Personal & Demographic Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    with c2:
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
        employment_status = st.selectbox(
            "Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"]
        )
    with c3:
        employer_cat = st.selectbox(
            "Employer Category",
            ["Private", "Government", "Unemployed", "MNC", "Business"],
        )

    st.subheader("2. Financial Information")
    c4, c5, c6 = st.columns(3)
    with c4:
        applicant_income = st.number_input(
            "Applicant Income ($)", min_value=0.0, value=10000.0, step=500.0
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income ($)", min_value=0.0, value=5000.0, step=500.0
        )
        savings = st.number_input(
            "Savings ($)", min_value=0.0, value=10000.0, step=500.0
        )
    with c5:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=675
        )
        dti_ratio = st.number_input(
            "DTI Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01
        )
        existing_loans = st.number_input(
            "Existing Loans Count", min_value=0, max_value=10, value=2
        )
    with c6:
        collateral_value = st.number_input(
            "Collateral Value ($)", min_value=0.0, value=25000.0, step=1000.0
        )

    st.subheader("3. Loan & Property Details")
    c7, c8, c9 = st.columns(3)
    with c7:
        loan_amount = st.number_input(
            "Loan Amount ($)", min_value=1000.0, value=20000.0, step=1000.0
        )
    with c8:
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84])
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"]
        )
    with c9:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit_btn = st.form_submit_button("Predict Loan Status")

if submit_btn:
    # 1. Reconstruct raw DataFrame matching training schema
    raw_input = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Employment_Status": employment_status,
                "Age": float(age),
                "Marital_Status": marital_status,
                "Dependents": float(dependents),
                "Credit_Score": float(credit_score),
                "Existing_Loans": float(existing_loans),
                "DTI_Ratio": float(dti_ratio),
                "Savings": savings,
                "Collateral_Value": collateral_value,
                "Loan_Amount": loan_amount,
                "Loan_Term": float(loan_term),
                "Loan_Purpose": loan_purpose,
                "Property_Area": property_area,
                "Education_Level": education,
                "Gender": gender,
                "Employer_Category": employer_cat,
            }
        ]
    )

    # 2. Impute Numeric & Categorical Features
    num_cols = raw_input.select_dtypes("number").columns
    cat_cols = raw_input.select_dtypes("object").columns

    raw_input[num_cols] = artifacts["num_imp"].transform(raw_input[num_cols])
    raw_input[cat_cols] = artifacts["cat_imp"].transform(raw_input[cat_cols])

    # 3. Ordinal Encoding
    raw_input["Education_Level"] = artifacts["oe"].transform(
        raw_input[["Education_Level"]]
    )

    # 4. One-Hot Encoding
    ohe_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category",
    ]

    ohe_encoded = artifacts["ohe"].transform(raw_input[ohe_cols])
    ohe_df = pd.DataFrame(
        ohe_encoded,
        columns=artifacts["ohe"].get_feature_names_out(ohe_cols),
        index=raw_input.index,
    )

    # Combine encoded columns and drop raw object columns
    processed_df = pd.concat([raw_input.drop(columns=ohe_cols), ohe_df], axis=1)

    # 5. Feature Transformations
    processed_df["Credit_Score"] = processed_df["Credit_Score"] ** 2
    processed_df["DTI_Ratio"] = processed_df["DTI_Ratio"] ** 2

    # Force exact column ordering as X_train
    processed_df = processed_df[artifacts["columns"]]

    # 6. Scaling
    scaled_input = artifacts["scaler"].transform(processed_df)

    # 7. Prediction
    pred = artifacts["model"].predict(scaled_input)[0]
    prob = artifacts["model"].predict_proba(scaled_input)[0][1]

    st.divider()

    if pred == 1:
        st.success(f"🎉 **Loan Approved!** Confidence: **{prob * 100:.1f}%**")
    else:
        st.error(f"❌ **Loan Rejected.** Approval Probability: **{prob * 100:.1f}%**")