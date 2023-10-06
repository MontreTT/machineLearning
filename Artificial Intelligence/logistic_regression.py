
import numpy as np
import math
import matplotlib.pyplot as plt

x = np.linspace(0,6, 20)
y = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]

p = lambda x : 1 / (1 + math.exp(-x))



likelihood = 0

for i in range(len(x)) :
    if y[i] == 1:
        likelihood -= math.log(p(i))
    else :
        likelihood -= math.log(1- p(i))
for i in range(len(x)):
    plt.plot(x[i], p(x[i]), 'bs')



print(likelihood)

plt.show()


