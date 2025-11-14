# 🏦 Loan Default Risk Prediction  
### IBM Machine Learning Project

This project is designed to predict whether a loan applicant is likely to **default** or **repay** a loan.  
Using basic financial and personal information, a Machine Learning model analyzes the risk level and helps make smarter lending decisions.

---

## 🎯 Project Objective

The main goal of this project is to build a simple and effective ML model that:

- Takes applicant financial details as input  
- Learns patterns of default vs non-default  
- Predicts whether a new applicant is **risky** or **safe**  

This kind of system is commonly used by banks, NBFCs, and financial institutions before approving loans.

---

## 📌 Why Loan Default Prediction?

Lending organizations face two major challenges:

1. Giving loans to high-risk customers → **Financial loss**  
2. Rejecting genuine customers → **Business loss**

A machine learning–based loan risk model helps solve both by:

- Reducing bad loans (defaults)  
- Improving approval accuracy  
- Making data-driven decisions  

---

## 📁 Dataset Description

The dataset consists of the following features:

| Feature | Description |
|--------|-------------|
| **income** | Applicant's yearly income |
| **loan_amount** | Loan amount requested |
| **credit_score** | Credit history score |
| **age** | Applicant’s age |
| **years_of_employment** | Total years of employment |
| **target** | 0 = No Default, 1 = Default |

The dataset is manually created using NumPy for demonstration purposes.

---

## 🧠 Machine Learning Model

**Algorithm Used:** Logistic Regression  

Reasons for choosing it:

- Ideal for binary classification (Default / No Default)  
- Simple and fast  
- Works well on small datasets  
- Easy to interpret  

### ML Workflow:

1. Create dataset  
2. Split into train & test sets  
3. Scale the features using StandardScaler  
4. Train Logistic Regression model  
5. Evaluate using accuracy and confusion matrix  
6. Predict risk for a new applicant  

---

## 🔧 Technologies Used

- Python  
- NumPy  
- Scikit-learn  
  - `train_test_split`  
  - `StandardScaler`  
  - `LogisticRegression`  
  - `accuracy_score`  
  - `confusion_matrix`  

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install numpy scikit-learn


###step2 python loan_default_prediction.py

