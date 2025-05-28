# EX-2 : IMPLEMENTATION OF SIMPLE LINEAR REGRESSION MODEL FOR PREDICTING THE MARKS SCORED 

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## PROGRAM:

```
Developed by: JAYASREE R
RegisterNumber: 212223040074
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head(10)
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
##plotting for test data
plt.scatter(x_test,y_test,color="grey")
plt.plot(x_test,y_pred,color="purple")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```


## OUTPUT:

#### Head value:

![image](https://github.com/user-attachments/assets/61964e83-f07e-43c2-a6c6-03c40586e1c4)

#### Tail Value:

![image](https://github.com/user-attachments/assets/bf495320-cf3b-4a5f-87a1-981abfee4ef8)

#### Hours Value:

![image](https://github.com/user-attachments/assets/c19401b4-2e41-4176-9622-6b365cd58e50)

#### Scores Value:

![image](https://github.com/user-attachments/assets/93aacbc4-9c81-4f4a-a5bb-e9e930dbce1a)

#### Y_Prediction:

![image](https://github.com/user-attachments/assets/9b711dbf-3537-41b7-93a0-64b7067dff17)

#### Y_Test:

![image](https://github.com/user-attachments/assets/ed2ed1ef-9dfa-4a96-bc7e-0f0414ee8cbf)

#### Result of MSE,MAE,RMSE:

![image](https://github.com/user-attachments/assets/8191ca04-0d04-4950-ba4b-c2c66399c959)

#### Training Set:

![image](https://github.com/user-attachments/assets/9c8aab62-cc2f-42ae-acc3-c67c53ef60e5)

#### Test Set:

![image](https://github.com/user-attachments/assets/e656a1eb-6108-4a72-9a06-8123bc27a654)









## RESULT:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
