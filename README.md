# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hycinth D
RegisterNumber:  212223240055
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:

![image](https://github.com/user-attachments/assets/47680b77-d33f-4ec3-8306-a269290ebc89)

![image](https://github.com/user-attachments/assets/c935d505-583d-4df0-a16d-f5b0154e26c4)

![image](https://github.com/user-attachments/assets/08737725-c0ae-4921-a893-547e96b8fa46)


![image](https://github.com/user-attachments/assets/cdba91e5-f0d0-4fc4-8687-63d5bbd405e8)

![image](https://github.com/user-attachments/assets/3b4a44e7-5a4a-4f5e-9f57-e306a65148e4)

![image](https://github.com/user-attachments/assets/8b70b227-13a5-4f43-9998-1761df22cb22)

![image](https://github.com/user-attachments/assets/a051c6a6-644f-4028-9518-65842de359f5)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
