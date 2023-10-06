import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../../Desktop/Happines Score Change from 2015 to 2017_2015-2017_data.csv', index_col = False)
def main():

    #get the happiness - gdp per person ,columns
    df2 = [df.iloc[:,2].to_numpy(),df.iloc[:,3].to_numpy()]

    k =3
    firstMeans =  meansInit(k , df2 )
    plt.scatter([i[0] for i in firstMeans], [i[1] for i in firstMeans], c="r")
    plt.scatter(df2[0], df2[1])
    plt.xlabel('Happiness')
    plt.ylabel('GDPR PER PERSON')
    plt.title("Iteration 0 " )
    plt.show()

    iteration = 1
    isFinished = False



    while isFinished == False :
        cluster = KMeans(firstMeans,k, df2)
        newmeans, isFinished = updateMeans(firstMeans, cluster)

        for j in range(k):
            plt.scatter([i[0] for i in cluster[j]], [i[1] for i in cluster[j]] )

        plt.scatter([i[0] for i in newmeans], [i[1] for i in newmeans], c="lime")

        plt.xlabel('Happiness')
        plt.ylabel('GDPR PER PERSON')
        plt.title("Iteration " + str(iteration))
        plt.show()
        iteration += 1





def KMeans(means,k,data): #k are the means_num , #data is an array that includes two arrays , happiness and gdp per person
    cluster = []  #array that includes all points , clustered in a mean
    for i in range(len(means)):
        a= []
        cluster.append(a)
    for i in range(k):
        cluster[i].append(means[i])
    for point in range(len(data[0])) : # for every value          #len(data[0])
        #print(data[0][point],data[1][point])
        distances = []

        for i in range(len(means)): #for every mean calculate each point distance then add rearrange the points to the means
            distance = math.sqrt(  (means[i][0]  -  data[0][point])**2  +  (means[i][1] - data[1][point])**2 )
            #print(distance)
            distances.append(distance)
        print(distances)
        print(np.array(distances).argmin())
        cluster[np.array(distances).argmin()].append([data[0][point],data[1][point]] )
        #print(cluster)

    return  cluster



def updateMeans(means , cluster) :
    '''
    flag = True
    while flag:
    '''
    oldmeans = means.copy()
    for i in range(len(means)):
        summx = 0
        summy = 0

        for j in cluster[i]:
            summx += j[0]
            summy += j[1]

        summ = [summx / len(cluster[i]), summy / len(cluster[i])]
        means[i] = summ

    for i in range(len(oldmeans)):
        #print(means[i], oldmeans[i])

        if abs(int(oldmeans[i][0]) - int(means[i][0])) + abs(int(oldmeans[i][1]) - int(means[i][1])) > (1/1000) * (abs(int(oldmeans[i][0])) +abs(int(oldmeans[i][1])))  :
            return means , False
        else:
            pass
    return means ,True






def meansInit( k , data):
    #means = [[4.1,0.6],[5.5,1.1],[7.0,1.7]]  # array that includes the means
    means = []
    counter = 0
    while counter != k : #initialize k points

        i = random.randint(0,len(data[0])-1)  # pointer to index of random value in data
        if str([data[0][i],data[1][i]]) not in means :
            means.append([data[0][i],data[1][i]])  # append random value as k point
            counter += 1
    return means







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
