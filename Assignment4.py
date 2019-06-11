import numpy as np
import pandas as pd
import math
import idx2numpy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# function to encode multiclass labels into binary format
def to_one_hot(Y):
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1.
    return binarized

# function to decode binary format to original format of class
def from_one_hot(Y):
    arr1 = np.zeros((len(Y), 1))
    for i in range(len(Y)):
        l = Y[i]
        for j in range(len(l)):
            if(l[j] == 1):
                arr1[i] = j+1
    return arr1

def softmax(X):
    "Numerically stable softmax function."
    exps = np.exp(X - np.max(X))
    return exps / exps.sum(axis=1, keepdims=True)

def Sigmoid(X):
    "Numerically stable sigmoid function."
    m = X.shape[0]
    n = X.shape[1]
    sig = np.ones((m,n))
    for i in range(m):
        for j in range(n):
            if X[i][j] >= 0:
                z = np.exp(-X[i][j])
                sig[i][j] = 1 / (1 + z)
            else:
                # if x is less than zero then z will be small, denominator can't be zero because it's 1+z.
                z = np.exp(X[i][j])
                sig[i][j] = z / (1 + z)
    return sig

def dSigmoid(x):
    s = Sigmoid(x)
    d_sig = s * (1-s)
    return d_sig

def main():
    x_train = idx2numpy.convert_from_file('/home/vinitkumar/Documents/Quizzes&Assgn/Assgn4/Samples/train-images-idx3-ubyte')
    y_trainA = idx2numpy.convert_from_file('/home/vinitkumar/Documents/Quizzes&Assgn/Assgn4/Samples/train-labels-idx1-ubyte')
    x_test = idx2numpy.convert_from_file('/home/vinitkumar/Documents/Quizzes&Assgn/Assgn4/Samples/t10k-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file('/home/vinitkumar/Documents/Quizzes&Assgn/Assgn4/Samples/t10k-labels-idx1-ubyte')

    M = x_train.shape[0]      # No. of instances
    N = x_train.shape[1]*x_train.shape[2]     # No. of attributes

    x_train.ravel()
    x_train = x_train.reshape(M,N)
    y_train = to_one_hot(y_trainA)
    print(y_trainA[5])
    print(y_train[5])

    labels = y_train.shape[1]
    L = 2    # No. of layers
    hnodes = 300  # No. of activation nodes in hidden layer
    loss = []
    lr = 0.3
    iter = 300

    #----declaration of weight and bias matrices----#
    np.random.seed(50)
    w0 = np.random.randn(N,hnodes)
    b0 = np.random.randn(hnodes)

    w1 = np.random.randn(hnodes,labels)
    b1 = np.random.randn(labels)

    for i in range(0, iter):
        # forward propagation
        #--------Layer 1------------#
        A0 = x_train
        Z1 = np.matmul(A0, w0) + b0
        A1 = Sigmoid(Z1)

        #--------Layer 2------------#
        Z2 = np.matmul(A1, w1) + b1
        A2 = softmax(Z2)

        # Back propagation
        #--------Layer 2------------#
        dloss_dZ2 = (A2 - y_train)
        dZ2_dw1 = A1

        dloss_dw1 = np.dot(dZ2_dw1.T, dloss_dZ2)
        dloss_db1 = dloss_dZ2

        #--------Layer 1------------#
        dloss_dA1 = np.dot((A2 - y_train), w1.T)
        dA1_dZ1 = dSigmoid(Z1)
        dZ1_dw0 = x_train

        dloss_dw0 = np.dot(dZ1_dw0.T, dA1_dZ1*dloss_dA1)
        dloss_db0 = dloss_dA1*dA1_dZ1

        #-----weight matrix update----#
        w1 -= lr*dloss_dw1
        b1 -= lr*dloss_db1.sum(axis=0)

        w0 -= lr*dloss_dw0
        b0 -= lr*dloss_db0.sum(axis=0)

        if i % 30 == 0:
            loss_ = np.sum(-y_train*np.log(A2))
            print ('Cost after iteration: ', i,' is ', loss_)
            loss.append(loss_)

    print('loss is: ', loss)
    #print('Accuracy is: ', (1-cost[-1])*100, '%')
    print('Weights from input to layer 1: ', w0)
    print('Weights from layer 1 to layer 2: ', w1)
    Y_predicted = from_one_hot(A2)

if __name__ == "__main__":
    main()
