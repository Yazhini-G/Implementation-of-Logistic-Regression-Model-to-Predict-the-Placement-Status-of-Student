# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:YAZHINI G

Register Number:212222220060  

import pandas as pd
df=pd.read_csv('/content/Placement_Data (1).csv')
df.head()

df1=df.copy()
df1=df1.drop(['sl_no','salary'],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear') #library for linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![266096983-1f27ecf4-7a8f-4728-9c4a-accd68757481](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/d43b41cd-316c-47b0-99d1-09c35864ceb9)
![266097007-fd429ba5-9693-4345-b062-0a8b64c49002](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/5a060431-c496-4423-8f3c-3d512a3c9a2a)
![266097057-1fd76e11-be2f-4594-b636-8bf73d6d4485](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/a8115816-6a3e-4e28-b3f1-f95cde0367b7)
![266097086-f64784c5-c9ad-4b9b-9d3f-f76bd832eafb](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/305e90a9-ee8f-4411-8c75-f102e477d7d8)
![266097109-9173ecdf-8efc-49f4-8fee-a8a7f8183dc6](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/08d26fb2-44ca-4141-b115-94318e86167a)
![266097133-c2d35385-a918-481b-bd84-6262fcc2ddbf](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/b36de68a-3a3d-49be-9efe-6b9778828c85)
![266097150-5561b69c-521d-4b76-b920-034a886a2f1f](https://github.com/Yazhini-G/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120244201/b7e08a74-9826-4494-a1da-9c4ebb7c4e27)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
