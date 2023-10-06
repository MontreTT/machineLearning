import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../../Desktop/Happines Score Change from 2015 to 2017_2015-2017_data.csv', index_col = False)

#get the happiness - gdp per person ,columns
df2 = df[["Happiness Score (2015-2017)", "Economy (GDP per Capita) (2015-2017)"]]

# rename columns
df2 = df2.rename(columns={"Happiness Score (2015-2017)": "Hapiness Score", "Economy (GDP per Capita) (2015-2017)": "GDP per Person"})
x_mean , y_mean = df2.mean(axis= 0)
print(x_mean , y_mean)
# calculating r , correlation coefficient
s_a = 0
s_x = 0

mean = 0


for i in range(len(df2)):
    mean += df2.iloc[i]["GDP per Person"]
    s_a += (df2.iloc[i]["Hapiness Score"] - x_mean) * (df2.iloc[i]["GDP per Person"] - y_mean)
    s_x += math.pow(df2.iloc[i]["Hapiness Score"] - x_mean,2)



mean = mean/ len(df2)

#calculating y ,x standard diviation


a = s_a / s_x




b = y_mean - (a * x_mean)


x = np.linspace(3.0, 8.0, num=30)



y =  a * x + b

plt.plot(x,y)
plt.scatter(df2["Hapiness Score"] , df2["GDP per Person"])

plt.show()