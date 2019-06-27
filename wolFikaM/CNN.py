import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.utils import shuffle
from utils import accuracy_rate, sparse_cross_entropy, draw_functions

EPSILON=np.float32(1e-37)

class Model:

	def __init__(self,layers,input_shape,num_classes):
		self.num_classes = num_classes
		self.batch_sz = input_shape[0]
		self.params = []
		self.layers = layers
		for ler in self.layers:
			self.params += ler.get_params()

		self.X = tf.placeholder(tf.float32, shape=input_shape)
		self.sparse_out = tf.placeholder(tf.int32,shape=self.batch_sz)

		#self.input = tf.placeholder(tf.float32, shape=(input_shape))
		#self.out = tf.nn.softmax(self.forward(self.input))

	def set_session(self,session):
		self.session = session
		init_op = tf.variables_initializer(self.params)
		self.session.run(init_op)

	def predict(self,X):
		assert(self.session is None)

		return self.session.run(
			self.out,
			feed_dict={self.input:X},
		)

	def forward(self,X,isTrain=True):
		for layer in self.layers:
			X = layer.forward(X,isTrain=isTrain)
		return X
	
	def fit_pure(self,Xtrain,Ytrain,Xtest,Ytest, optimizer= None, epoch=1,test_period=1,show_figure=False):

		assert(optimizer is not None)
		assert(self.session is not None)

		try:
			Xtrain = Xtrain.astype(np.float32)
			Xtest = Xtest.astype(np.float32)

			Yish = self.forward(self.X,isTrain=True)
			cost = (
				tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yish, labels=self.sparse_out)), Yish,
			)
			train_op = (cost,optimizer.minimize(cost[0]))
			optimizer.init_variables(self.session)

			Yish_test = tf.nn.softmax(self.forward(self.X,isTrain=False))

			n_batches=Xtrain.shape[0] // self.batch_sz

			train_costs=[]
			train_accuracys=[]
			test_costs=[]
			test_accuracys=[]

			for m in range(epoch):
				Xtrain, Ytrain = shuffle(Xtrain,Ytrain)
				train_cost=np.float32(0)
				train_accuracy=np.float32(0)
				iterator = range(n_batches)

				for i in tqdm(iterator):
					Xbatch = Xtrain[i*self.batch_sz:(i+1)*self.batch_sz]
					Ybatch = Ytrain[i*self.batch_sz:(i+1)*self.batch_sz]

					(train_cost_batch,y_ish),_ = self.session.run(
						train_op,
						feed_dict={self.X:Xbatch,self.sparse_out:Ybatch},
					)
					# Use exponential decay for calculating loss and error
					train_cost = 0.99 * train_cost + 0.01 * train_cost_batch
					train_accur_batch = accuracy_rate(np.argmax(y_ish, axis=1), Ybatch)
					train_accuracy = 0.99 * train_accuracy + 0.01 * train_accur_batch

				# Validating the network on test data
				if m % test_period == 0:
					# For test data
					test_cost = np.float32(0)
					test_predictions = np.zeros(len(Xtest))

					for k in range(len(Xtest) // self.batch_sz):
						# Test data
						Xtestbatch = Xtest[k * self.batch_sz:(k + 1) * self.batch_sz]
						Ytestbatch = Ytest[k * self.batch_sz:(k + 1) * self.batch_sz]
						Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
						test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
						test_predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_test_done, axis=1)

					# Collect and print data
					test_cost = test_cost / (len(Xtest) // self.batch_sz)
					test_accuracy = accuracy_rate(test_predictions, Ytest)
					test_accuracys.append(test_accuracy)
					test_costs.append(test_cost)

					train_costs.append(train_cost)
					train_accuracys.append(train_accuracy)

					print('\tEpoch:', (m+1), 'Train accuracy: {:0.4f}'.format(train_accuracy), 'Train cost: {:0.5f}'.format(train_cost),
					  'Test accuracy: {:0.4f}'.format(test_accuracy), 'Test cost: {:0.5f}'.format(test_cost))
					#need add plot by pyplot (i will understand!)
			if show_figure:
				draw_functions({'Train cost':train_costs, 'Test cost':test_costs})
				draw_functions({'Train accuracy':train_accuracys,'Test accuracy':test_accuracys})

		except Exception as ex:
			print(ex)
			iterator.close()
		finally:
			return {'train costs': train_costs, 'train errors': train_accuracys,
				'test costs': test_costs, 'test errors': test_accuracys}           
