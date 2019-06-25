import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import error_rate, y2indicator, cifar10_test

import time
from sklearn.utils import shuffle

def convpool(X,W,b):
	conv_out = tf.nn.conv2d(X,W, strides=[1,1,1,1], padding='SAME')
	conv_out = tf.nn.bias_add(conv_out, b)
	pool_out = tf.nn.max_pool(conv_out, ksize =[1,2,2,1], strides = [1,2,2,1], padding='SAME')

	return tf.nn.relu(pool_out)

def init_filter(shape,poolsz):
	W = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2] / np.prod(poolsz)))
	return W.astype(np.float32)

def rearange(X):#we need (N,32,32,3) but we got 
	return (X.transpose(0, 3, 1, 2) / 255).astype(np.float32)



def main():
	#some load data
	K = 10
	(Xtrain,Ytrain),(Xtest,Ytest) = cifar10_test()
	Xtrain = (Xtrain / 255).astype(np.float32)
	Xtest = (Xtest / 255).astype(np.float32)
	Ytrain_ind = y2indicator(Ytrain,K).astype(np.int32)
	Ytest_ind = y2indicator(Ytest,K).astype(np.int32)

	print(Xtrain.shape)
	print(Ytrain_ind.shape)
	epoch = 200
	print_period = 10
	N = Xtrain.shape[0]
	batch_sz = 250
	n_batches = N // batch_sz
	n_batches_test = Xtrain.shape[0] // batch_sz

	M = 512
	M1 = 512
	#K = 10 ABOVE
	poolsz = (2,2)

	W1_shape = (5,5,3,16)#32 / 2 = 16
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

	W2_shape = (5,5,16,40)# 16 / 2= 8
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[-1], dtype = np.float32)


	W3_shape = (5,5,40,100)# 8 / 2 =4
	W3_init = init_filter(W3_shape, poolsz)
	b3_init = np.zeros(W3_shape[-1], dtype = np.float32)


	W4_shape = (5,5,100,196)#4 / 2 = 2
	W4_init = init_filter(W4_shape, poolsz)
	b4_init = np.zeros(W4_shape[-1], dtype = np.float32)


	W5_init = np.random.randn(W4_shape[-1]*2*2, M) / np.sqrt(W4_shape[-1]* 2 * 2 + M)
	b5_init = np.zeros(M,dtype=np.float32)
	W6_init = np.random.randn(M,M1) / np.sqrt(M+M1)
	b6_init = np.zeros(M1,dtype=np.float32)
	W7_init = np.random.randn(M1,K) / np.sqrt(M1 + K)
	b7_init = np.zeros(K,dtype=np.float32)

	X = tf.placeholder(tf.float32, shape = (batch_sz,32,32,3), name = 'X')
	T = tf.placeholder(tf.float32, shape = (batch_sz, K), name = 'T')

	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))

	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))

	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))

	W4 = tf.Variable(W4_init.astype(np.float32))
	b4 = tf.Variable(b4_init.astype(np.float32))
	
	W5 = tf.Variable(W5_init.astype(np.float32))
	b5 = tf.Variable(b5_init.astype(np.float32))

	W6 = tf.Variable(W6_init.astype(np.float32))
	b6 = tf.Variable(b6_init.astype(np.float32))

	W7 = tf.Variable(W7_init.astype(np.float32))
	b7 = tf.Variable(b7_init.astype(np.float32))

	Z1 = convpool(X,W1,b1)
	Z2 = convpool(Z1,W2,b2)
	Z3 = convpool(Z2,W3,b3)
	Z4 = convpool(Z3,W4,b4)
	Z4_shape = Z4.get_shape().as_list()

	Z4r = tf.reshape(Z4,[-1, np.prod(Z4_shape[1:])])
	Z5 = tf.nn.relu(tf.matmul(Z4r,W5) + b5)
	Z6 = tf.nn.relu(tf.matmul(Z5,W6) + b6)
	Yish = tf.matmul(Z6,W7) + b7

	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish,labels=T))
	cost_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish,labels=T))

	train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

	predict_op = tf.argmax(tf.nn.softmax(Yish), axis=1)

	t0 = time.time()
	LL = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(epoch):
			for j in range(n_batches):

				Xbatch = Xtrain[j*batch_sz:batch_sz*(j+1)]
				Ybatch = Ytrain_ind[j*batch_sz:batch_sz*(j+1)]

				if len(Xbatch) == batch_sz:
					session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
					if j % print_period == 0:
						# due to RAM limitations we need to have a fixed size input
						# so as a result, we have this ugly total cost and prediction computation
						test_cost = 0
						prediction = np.zeros(len(Xtest))
						for k in range(Xtest.shape[0] // batch_sz):
							Xtestbatch = Xtest[k*batch_sz:batch_sz*(k+1)]
							Ytestbatch = Ytest_ind[k*batch_sz:batch_sz*(k+1),]
							test_cost += session.run(cost_test, feed_dict={X: Xtestbatch, T: Ytestbatch})
							prediction[k*batch_sz:batch_sz*(k+1)] = session.run(
								predict_op, feed_dict={X: Xtestbatch})
						accur = error_rate(prediction, np.argmax(Ytest_ind,axis=1))
						print(f'epoch is {i} accuracy is {round(accur,5)} and cost is {round(test_cost,5)}')
						LL.append(test_cost)

	plt.plot(LL)
	plt.show()

if __name__ == '__main__':
	main()