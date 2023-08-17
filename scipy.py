from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
y=[]
x=np.random.uniform(5,100,100)
for i in x:

    j=abs(0.2*i-np.random.randint(5))
    y.append(j)

res = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.show()
