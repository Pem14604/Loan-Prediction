# Loan-Prediction

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. 

It is classification problem and in this we have following variables.

INDEPENDENT VARIABLE
Loan_ID               object
Gender                object
Married               object
Dependents            object
Education             object
Self_Employed         object
ApplicantIncome        int64
CoapplicantIncome    float64
LoanAmount           float64
Loan_Amount_Term     float64
Credit_History       float64
Property_Area         object

DEPENDENT VARIABLE
Loan_Status           object

I used Random Forest, Logistic Regression, SVM, Decision Tree, XGboost to compare the performance and found Logistic Regression doing very well as comparision to other and have accuray 83 %

