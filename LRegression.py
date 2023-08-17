import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
y=[]
x=np.random.uniform(5,100,100)
for i in x:

    j=abs(0.2*i-np.random.randint(5))
    y.append(j)

model=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
x_train=x_train.reshape(-1,1)
model.fit(x_train,y_train)
x_test=x_test.reshape(-1,1)
y_pred=model.predict(x_test)

plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.legend(['x_test,y_test_scatter','x_test,y_predict_plot'])

plt.xticks(())
plt.yticks(())

plt.show()
# plt.scatter(x,y)
# plt.show()
# print(y)