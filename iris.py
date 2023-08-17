import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix

a=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\iris.csv')

x=a.drop(['Species','Id'],axis=1)
y=a['Species']


#y=y.replace(['Iris-setosa' ,'Iris-versicolor' ,'Iris-virginica'],[1,2,3])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

w= [1,3,5,7,9,11,13]
ww=[]
for k in w:
 model=KNeighborsClassifier(k)

 model.fit(x_train,y_train)
 y_pred=model.predict(x_test)
 s=model.score(x_train,y_train)
 ww.append(s)

 confusion=confusion_matrix(y_test,y_pred)

print(confusion)
plt.imshow(confusion)

# plt.plot(w,ww,c='r')
# plt.xlabel('K',fontsize=16)
# plt.ylabel('Score',fontsize=16)
# plt.title('Score_Versus_K_Factor_for_Iris_Dataset',fontsize=16)
plt.show()
