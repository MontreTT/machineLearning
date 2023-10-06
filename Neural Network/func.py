import numpy as np
import pandas as pd

def softmax( x, ax=1 ):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x -m )
    return ( p / np.sum(p,axis=ax, keepdims=True) )


def cost(w1, w2,  y, y_train, lamda):

    E = np.sum(y_train* np.log(y+1e-9))
    E -= lamda * np.sum(np.square(w1)) / 2 + lamda * np.sum(np.square(w2)) / 2

    return E

def activationFuncLog(a):
    h = np.log(1 + np.exp(a))
    return h

def activationFuncHype(a):
    h = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
    return h
def activationFuncCos(a):
    h = np.cos(a)
    return h
def actFuncDerLog(a):
    h = np.exp(a)/(1 + np.exp(a) )
    return h

def actFuncDerHype(a):

    h = (4 * np.exp(2 * a)) / (1 + np.exp(2 * a)**2)
    return h

def actFunDerCos(a):
    h = - np.sin(a)
    return h

def w1Der(a, h, y , t , w2 , ):
    return (y-t) * w2 * h(a)


def load_data():
    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """

    # load the train files
    df = None

    y_train = []

    for i in range(10):
        tmp = pd.read_csv('mnistdata/train%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_train.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    train_data = df.to_numpy()
    y_train = np.array(y_train)

    # load test files
    df = None

    y_test = []

    for i in range(10):
        tmp = pd.read_csv('mnistdata/test%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector

        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_test.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    test_data = df.to_numpy()
    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test