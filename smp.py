import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("student_info.csv")
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.model_selection import train_test_split
#x=data.head()
#print(x)
#y=data.describe()
#print(y)
#z=data.info()
#print(z)


"""m=data['study_hours']
n=data['student_marks']
plt.scatter(m,n)
plt.xlabel("study hours")
plt.ylabel("student marks")
plt.title("student marks predication")
plt.show()"""

#o=data.isnull().sum()
#print(o)
newdata=data.fillna(data.mean())
#print(newdata.head(20))

x=newdata[['study_hours']]
y=newdata[['student_marks']]
#x=print(newdata.iloc[:,0].values)
#y=print(newdata.iloc[:,1].values)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
#print(lr.predict([[4]])[0][0].round(2))
#print(lr.score(x_test,y_test))


import pickle
pickle.dump(lr,open("student_marks_predictor_model.pkl",'wb'))
model=pickle.load(open("student_marks_predictor_model.pkl",'rb'))


#print(model.predict([[5]])[0][0].round(2))




