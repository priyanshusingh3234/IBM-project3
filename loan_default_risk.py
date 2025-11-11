# ==================================================
# IBM PROJECT: Loan Default Risk Prediction
# Author: Priyanshu Singh
# ==================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# --------------------------------------------------
# STEP 1: Dataset Creation
# --------------------------------------------------
# Format: [income, loan_amount, credit_score, age, years_of_employment]
# Target: 0 = No Default, 1 = Default

X = np.array([
    [40000, 10000, 700, 25, 2],
    [25000, 12000, 580, 22, 1],
    [60000, 20000, 720, 35, 5],
    [32000, 15000, 610, 28, 2],
    [80000, 25000, 750, 40, 10],
    [22000, 10000, 590, 23, 1],
    [90000, 30000, 780, 45, 12],
    [30000, 15000, 600, 26, 2],
    [100000, 35000, 790, 50, 15],
    [28000, 8000, 620, 24, 1]
])

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# --------------------------------------------------
# STEP 2: Split Data (Training + Testing)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# STEP 3: Scale the Data
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# STEP 4: Train Logistic Regression Model
# --------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# STEP 5: Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# STEP 6: Evaluation
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===============================")
print(" Loan Default Risk Prediction ")
print("===============================")
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("📊 Confusion Matrix:")
print(cm)

# --------------------------------------------------
# STEP 7: Predict for New Applicant
# --------------------------------------------------
new_applicant = np.array([[45000, 12000, 680, 27, 3]])  # sample input
new_applicant_scaled = scaler.transform(new_applicant)
prediction = model.predict(new_applicant_scaled)

print("\n🔍 Prediction for new applicant:")
if prediction[0] == 1:
    print("❌ This applicant is likely to DEFAULT on the loan.")
else:
    print("✅ This applicant is likely to REPAY the loan.")

# --------------------------------------------------
# STEP 8: Project Complete
# --------------------------------------------------
print("\n🎯 Project Execution Completed Successfully!")