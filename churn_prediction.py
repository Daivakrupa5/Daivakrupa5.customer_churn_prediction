# churn_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lifelines import CoxPHFitter

# Load dataset
df = pd.read_csv("sample_churn_dataset.csv")

# Basic EDA
print("First 5 rows:\n", df.head())
print("\nSummary:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # Male=1, Female=0
df['Partner'] = le.fit_transform(df['Partner'])  # Yes=1, No=0
df['Dependents'] = le.fit_transform(df['Dependents'])
df['PhoneService'] = le.fit_transform(df['PhoneService'])
df['MultipleLines'] = le.fit_transform(df['MultipleLines'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['OnlineSecurity'] = le.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = le.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = le.fit_transform(df['DeviceProtection'])
df['TechSupport'] = le.fit_transform(df['TechSupport'])
df['StreamingTV'] = le.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = le.fit_transform(df['StreamingMovies'])
df['Contract'] = le.fit_transform(df['Contract'])
df['PaperlessBilling'] = le.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID if present
df = df.drop(['customerID'], axis=1, errors='ignore')

# Split data
X = df.drop(['Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance plot
plt.figure(figsize=(10,6))
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Survival Analysis using Cox Proportional Hazards model
df_survival = df[['tenure', 'Churn'] + [col for col in df.columns if col not in ['Churn', 'tenure']]]
cox_df = df_survival.copy()
cox_df['event'] = df_survival['Churn']  # 1 if churned, 0 otherwise

cox = CoxPHFitter()
cox.fit(cox_df, duration_col='tenure', event_col='event')

print("\nCox Proportional Hazards Summary:\n")
cox.print_summary()