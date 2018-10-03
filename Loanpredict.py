

import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
import sklearn
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.dtypes


train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)



test.isnull().sum()
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)


x = train.drop('Loan_Status', axis=1)
y = train.Loan_Status
x=x.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID', axis=1)



x=pd.get_dummies(x)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


from sklearn.model_selection import train_test_split
x_train, x_cv,y_train,y_cv =train_test_split(x,y,test_size=0.3, random_state=0 )

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_cv=sc.fit_transform(x_cv)
test=sc.fit_transform(test)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator= lr, X=x_cv, y=y_cv, cv=10)
accuracies.mean()
accuracies.std()
Pred_test=lr.predict(test)


Submission=pd.read_csv('sub.csv')
Submission['Loan_Status']=Pred_test
Submission['Loan_Status'].replace(0, 'N',inplace=True)
Submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(Submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')
