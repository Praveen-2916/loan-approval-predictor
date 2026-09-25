import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv("data/loan_approval_data.csv")
df = df.drop("Applicant_ID", axis=1)

X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X_train.select_dtypes("number").columns
cat_cols = X_train.select_dtypes("object").columns

num_imp = SimpleImputer(strategy="mean")
X_train[num_cols] = num_imp.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imp.transform(X_test[num_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
X_train[cat_cols] = cat_imp.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imp.transform(X_test[cat_cols])

y_train = y_train.fillna(y_train.mode().iloc[0])
y_test = y_test.fillna(y_train.mode().iloc[0])

oe = OrdinalEncoder(categories=[["Not Graduate", "Graduate"]])
X_train["Education_Level"] = oe.fit_transform(X_train[["Education_Level"]])
X_test["Education_Level"] = oe.transform(X_test[["Education_Level"]])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

X_train_encoded = ohe.fit_transform(X_train[cols])
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out(cols), index=X_train.index)

X_test_encoder = ohe.transform(X_test[cols])
X_test_encoded_df = pd.DataFrame(X_test_encoder, columns=ohe.get_feature_names_out(cols), index=X_test.index)

X_train = pd.concat([X_train.drop(columns=cols), X_train_encoded_df], axis=1)
X_test = pd.concat([X_test.drop(columns=cols), X_test_encoded_df], axis=1)

X_train["Credit_Score"] = X_train["Credit_Score"] ** 2
X_train["DTI_Ratio"] = X_train["DTI_Ratio"] ** 2

X_test["Credit_Score"] = X_test["Credit_Score"] ** 2
X_test["DTI_Ratio"] = X_test["DTI_Ratio"] ** 2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# Evaluation metrics
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")

# =========================================================
# SAVE ALL FIT PREPROCESSORS & MODEL FOR DEPLOYMENT
# =========================================================
joblib.dump(num_imp, "num_imp.pkl")
joblib.dump(cat_imp, "cat_imp.pkl")
joblib.dump(oe, "oe.pkl")
joblib.dump(ohe, "ohe.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(log_model, "log_model.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

print("\nSuccessfully saved all 7 deployment artifacts!")