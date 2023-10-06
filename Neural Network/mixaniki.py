import numpy as np
import pandas as pd
import func as f
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle




def accuracy(y, t):
    return np.argmax(y, axis=1), np.argmax(t, axis=1)





def forwardRep(x, h, w1, w2):
    o1 = x @ w1
    z = h(o1)
    o2 = z @ w2
    output = f.softmax(o2)
    return output, z, o1


def backProp(error, w1, w2, hd, x, z, o1, lamda, lr):
    dw1 = x.T @ ((error @ w2.T) * hd(o1)) - lamda * w1
    dw2 = z.T @ error - lamda * w2
    w1 += lr * dw1
    w2 += lr * dw2
    return w1, w2

def main (lr , lamda , M , n_epochs , datatype , activationfunc):
    activationlist = [f.activationFuncLog , f.activationFuncHype, f.activationFuncCos,f.actFuncDerLog,f.actFuncDerHype,f.actFunDerCos]


    X_train, X_test, y_train, y_test = f.load_data()
    plt.close('all')
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    print(X_train.shape, y_train.shape)

    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255
    # X_train = np.hstack((np.ones((X_train.shape[0],1) ), X_train))
    # X_test = np.hstack((np.ones((X_test.shape[0],1) ), X_test) )

    N, D = X_train.shape  # total samples , features (60.000 χ 784)
    K = 10  # K_means
    batchSize = 100  # size of batches
    batchNum = X_train.shape[0] // batchSize  # Num of batchses

    w1 = np.random.normal(loc=0.0, scale=1.0, size=(D, M)) * 0.01  # Weight1 (features, M  )(784 , M)
    w2 = np.random.normal(loc=0.0, scale=1.0, size=(M, K)) * 0.01  # Weight2 (M ,  K)(M,10)


    h = activationlist[activationfunc]  #activation func :log ,  hype  or cos
    hd = activationlist[activationfunc + 3]
    costs = []
    acc = []
    for epoch in range(n_epochs):
        lr = lr / 1.8 #learning rate update per epoch (decay)
        for i in range(batchNum):
            x = X_train[i * batchSize: (i + 1) * batchSize, :]  #batchsize of x
            t = y_train[i * batchSize: (i + 1) * batchSize, :]
            y, z, o1 = forwardRep(x, h, w1, w2)  # [batchNum x K_means  ]
            w1, w2 = backProp(t - y, w1, w2, hd, x, z, o1, lamda, lr)
            costs.append(f.cost(w1, w2, y, t, lamda))
        c1, c2 = accuracy(y, t)
        acc.append(accuracy_score(c1, c2))
        print("Number of epoch :  " + str(epoch))

    plt.figure()
    plt.legend("acc/epoch")
    plt.plot(acc)
    plt.figure()
    plt.legend("cost/batches")
    plt.plot(costs)
    plt.show()
    x = X_test
    ys = y_test
    y, z, o1 = forwardRep(x, h, w1, w2)

    c1, c2 = accuracy(y, ys)
    print()
    print("data type are: " + str(datatype))
    print("M hidden Units are: " + str(M))
    print("learing rate is: " + str(lr))
    print("lamda is: " + str(lamda))
    act = ["log" , "Hype", "cos"]
    print("activation func is: " + act[activationfunc])
    print("Total accuracy score for test Data is :  " + str(accuracy_score(c1, c2) * 100) + "%")


main(lr =0.005 , lamda = 0.0001 ,M = 100 , n_epochs= 10 , datatype="mnist-data" ,activationfunc= 2)