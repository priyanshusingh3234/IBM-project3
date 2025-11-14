# 🏦 Loan Default Risk Prediction  
### IBM Machine Learning Project

This project focuses on predicting whether a loan applicant is likely to **default** or **repay** a loan using a Machine Learning model. The goal is to help financial institutions make safer and more accurate lending decisions by analyzing key applicant information.

---

## 📘 Introduction

Loan default prediction is an essential task in the finance industry. Banks and lending companies need to understand which applicants are safe to lend money to and which may cause future financial risk.  
Machine Learning models help automate this process by learning patterns from past data and predicting the probability of default for new applicants.

This project demonstrates a simple, easy-to-understand ML pipeline built using Python and Scikit-learn.

---

## 🎯 Objective of the Project

- Build a logistic regression model for predicting loan default risk  
- Train the model using financial and personal details of applicants  
- Evaluate the model using accuracy and confusion matrix  
- Predict the outcome for a new applicant  
- Demonstrate how ML can support real-world loan decision systems  

---

## 📁 Dataset Description

The dataset is created manually using NumPy and contains five input features:

| Feature | Description |
|--------|-------------|
| **income** | Annual income of the applicant |
| **loan_amount** | Total loan amount requested |
| **credit_score** | Applicant’s credit score |
| **age** | Age of the applicant |
| **years_of_employment** | Total years of employment |
| **target** | 0 = No Default, 1 = Default |

This dataset helps the model identify patterns between applicant attributes and their repayment behavior.

---

## 🧠 Machine Learning Model Used

### **Logistic Regression**

This algorithm is used because:

- It is ideal for binary classification (Default / No Default)  
- It works well on small datasets  
- Training is fast and results are easy to interpret  
- It is widely used in financial risk prediction  

### Workflow:

1. Create dataset  
2. Train-test split  
3. Standardize features  
4. Train Logistic Regression model  
5. Predict results  
6. Evaluate model  
7. Predict risk for a new applicant  

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Scikit-learn  
  - LogisticRegression  
  - StandardScaler  
  - train_test_split  
  - accuracy_score  
  - confusion_matrix  

---

