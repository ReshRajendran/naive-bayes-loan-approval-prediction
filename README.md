# Loan Approval Prediction using Naive Bayes

A machine learning project that predicts whether a loan application will be approved, using the Naive Bayes algorithm.

## 📌 Overview
This project uses applicant data to predict loan approval status (**Status**: Y/N) based on features like gender, marital status, dependents, education, employment type, income, loan amount, credit history, and property area.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (BernoulliNB, LabelEncoder, StandardScaler, train_test_split)

## 🔍 Workflow
1. Loaded and explored the loan dataset (614 rows, 12 columns)
2. Handled missing values across Gender, Married, Dependents, Self_Employed, Credit_History, and Term using mode imputation
3. Encoded categorical features (Gender, Married, Dependents, Education, Self_Employed, Area, Status) using LabelEncoder
4. Split data into training and testing sets (70/30)
5. Scaled features using StandardScaler
6. Trained a Bernoulli Naive Bayes classifier
7. Evaluated performance using accuracy score

## 📊 Results
- **Accuracy: 78.38%**

## 📁 Dataset
Loan applicant dataset with features: Gender, Married, Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income, Loan_Amount, Term, Credit_History, Area, and Status.

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: `pip install numpy pandas scikit-learn`
3. Open the notebook and run all cells

