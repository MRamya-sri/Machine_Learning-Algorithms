import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('D:\\DATA SCIENCE\\MACHINE LEARNING\\Simple Linear Regression\\')
os.getcwd()
df1= pd.read_csv('Salary_Data.csv')
df1
#x independent variable
x= df1.iloc[:,:-1].values
x
#y dependent variables
y= df1.iloc[:,1].values
y
#creating a plot to check the trend
plt.plot(x,y)
plt.show()
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
print('x shape',x.shape)
print('y shape',y.shape)
print('x_train shape',x_train.shape)
print('x_test shape',x_test.shape)
print('y_train shape',y_train.shape)
print('y_test shape',y_test.shape)
#model fitting

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)
print(lr)
#prediction y=mx+c y=prediction m=coefficient x=experience c = intercept

y_pred = lr.predict(x_test)
print(y_pred)

#y= mx +c (Coefficient and Interceptor Values) 
#Y= slope
print ('Coefficient', lr.coef_)
print ('Intercept', lr.intercept_)

#display the prediction with difference

df_x_test = pd.DataFrame(x_test, columns=['Experience'])
df_y_test = pd.DataFrame(y_test, columns=['Salary'])
df_y_test_pred =pd.DataFrame(y_pred, columns=['Prediction'])
df_diff = df_y_test-df_y_test_pred
y_test_pred = pd.concat([df_x_test ,df_y_test, df_y_test_pred ],axis =1)
y_test_pred['Difference']= df_y_test ['Salary']- df_y_test_pred['Prediction']
print(y_test_pred)

#accuracy of model
from sklearn.metrics import r2_score
accuracy = r2_score(y_test,y_pred)
print(accuracy)

#test data
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,'r')
plt.show()

#now prediction of complete data
y_pred_final = lr.predict(x)
print(y_pred_final)

#final result in dataframe storing in excel

y_pred_final = pd.DataFrame(y_pred_final,columns=['Prediction'])
result=pd.concat([df1,y_pred_final],axis=1)
result['Difference']=result['Salary']-result['Prediction']
print(result)
result.to_excel('D:\\DATA SCIENCE\\MACHINE LEARNING\\Simple Linear Regression\\prediction.xlsx')

from sklearn.metrics import r2_score
accuracy=r2_score(y,y_pred_final)
print(accuracy)

plt.plot(x,y)
plt.plot(x,y_pred_final,'red')
plt.show()
