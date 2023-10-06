import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)
z= [5,5,5,5,5,5,5,4,5,5]
plt.scatter(x,y, c = z)
plt.show()
print(type(z))
print(z)


