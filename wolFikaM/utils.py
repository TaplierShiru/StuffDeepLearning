import numpy as np
from tensorflow.keras.datasets import mnist as MN
from tensorflow.keras.datasets import cifar10

def relu(a):
	return a * ( a > 0)

def accuracy_rate(tar,pred):
	return np.mean(tar == pred)

def y2indicator(y,K):
	N = len(y)
	ind = np.zeros((N,K))

	for i in range(N):
		ind[i,y[i]] = 1

	return ind.astype(np.float32)

def test_mnist():
	(Xtrain,Ytrain), (Xtest, Ytest) = MN.load_data()

	return (Xtrain,Ytrain),(Xtest,Ytest)

def cifar10_test():
	return cifar10.load_data()

def sparse_cross_entropy(Y, T):
    return -np.mean(np.log(Y[np.arange(Y.shape[0]), T]))