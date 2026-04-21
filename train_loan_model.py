import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("loan.csv")
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# Preprocessing
df = df.drop('Loan_ID', axis=1)

# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Fix Dependents
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Encode target
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Feature Engineering
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Per_Dependent'] = df['Total_Income'] / (df['Dependents'] + 1)
df['Loan_to_Income_Ratio'] = (df['LoanAmount'] * 1000) / df['Total_Income']

# EMI Calculation (10% annual interest)
r = 0.10 / 12
n = df['Loan_Amount_Term']
P = df['LoanAmount'] * 1000
df['EMI'] = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 12)
df['Credit_Category'] = pd.cut(df['Credit_History'], bins=[-1, 0.5, 1], labels=['Poor', 'Good'])

# Encode categoricals
le_dict = {}
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_Category']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Save encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

# Features
feature_cols = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area', 'Total_Income', 'Income_Per_Dependent',
    'Loan_to_Income_Ratio', 'EMI', 'Balance_Income', 'Credit_Category'
]

X = df[feature_cols]
y = df['Loan_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel and encoders saved successfully!")