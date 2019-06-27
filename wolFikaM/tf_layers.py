import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def custom_softmax(x):
	m = tf.reduce_max(x,1)
	x = x - m
	e = tf.exp(x)
	return e / tf.reduce_sum(e,-1)


def init_filter(d,f_in,f_out,stride):
	return (np.random.randn(d,d,f_in,f_out) * np.sqrt(2.0 / (d * d * f_in))).astype(np.float32)
	
class Layer:

	def __init(self):
		self.params = []

	def forward(self,X,isTrain=True):
		pass

	def get_params(self):
		return self.params


class ConvLayer(Layer):

	def __init__(self,d,f_in,f_out,stride=1,padding='SAME'):
		self.W = tf.Variable(init_filter(d,f_in,f_out,stride))
		self.b = tf.Variable(np.zeros(f_out,dtype=np.float32))
		self.stride=stride
		self.padding = padding
		self.params = [self.W, self.b]


	def forward(self,X,isTrain=True):
		X = tf.nn.conv2d(
			X,
			self.W,
			strides=[1,self.stride,self.stride,1],
			padding=self.padding,
		)
		return X + self.b

	def copyFromKerasLayers(self,layer):
		W,b = layer.get_weights()
		op1=self.W.assign(W)
		op2=self.b.assign(b)
		#self.session.run((op1,op2))


class BatchNormLayer(Layer):

	def __init__(self,D):
		self.running_mean=tf.Variable(np.zeros(D,dtype=np.float32),trainable=False)
		self.running_var=tf.Variable(np.ones(D,dtype=np.float32),trainable=False)
		self.gamma = tf.Variable(np.ones(D,dtype=np.float32))
		self.beta=tf.Variable(np.zeros(D,dtype=np.float32))

		self.params = [self.running_mean, self.running_var, self.gamma, self.beta]

	def forward(self,X,isTrain=True,decay=0.9):
		axes=None
		if len(X.shape) == 4:
			axes = [0,1,2]
		else:
			axes = [0]

		if isTrain:
			batch_mean, batch_var = tf.nn.moments(X, axes=axes)
			update_running_mean = tf.assign(
				self.running_mean,
				self.running_mean * decay + batch_mean * (1 - decay),
			)
			update_running_var = tf.assign(
				self.running_var,
				self.running_var * decay + batch_var * (1 - decay),
			)

			with tf.control_dependencies([update_running_mean, update_running_var]):
				out = tf.nn.batch_normalization(
				  X,
				  batch_mean,
				  batch_var,
				  self.beta,
				  self.gamma,
				  1e-4
				)
		else:
			out = tf.nn.batch_normalization(
				X,
				self.running_mean,
				self.running_var,
				self.beta,
				self.gamma,
				1e-4
			)
		return X

	def copyFromKerasLayers(self,layer):
		gamma,beta,running_mean,running_var=layer.get_weights()
		op1 = self.running_mean.assign(running_mean)
		op2 = self.running_var.assign(running_var)
		op3=self.gamma.assign(gamma)
		op4=self.beta.assign(beta)
		#self.session.run((op1,op2,op3,op4))

class ActivationLayer:
	def __init__(self,activation=tf.nn.relu):
		self.activation=activation

	def forward(self, X,isTrain=True):
		return self.activation(X)

	def get_params(self):
		return []

class ActivLeakyRelu(Layer):
	def __init__(self,alpha=0.2):
		self.alpha = alpha

	def forward(self,X,isTrain=True):
		return tf.nn.leaky_relu(X,alpha=self.alpha)

	def get_params(self):
		return []

class MaxPoolLayer(Layer):
	def __init__(self, dim=2,strides=2,padding='SAME'):
		self.dim = dim
		self.strides = strides
		self.padding=padding

	def forward(self, X,isTrain=True):
		return tf.nn.max_pool(
		  X,
		  ksize=[1, self.dim, self.dim, 1],
		  strides=[1, self.strides, self.strides, 1],
		  padding=self.padding,
		)

	def get_params(self):
		return []

class AvgPool(Layer):

	def __init__(self, ksize=1,padding='SAME',strides=1):
		self.ksize = ksize
		self.strides = strides
		self.padding=padding

	def forward(self,X,isTrain=True):
		return tf.nn.avg_pool(
			X,
			ksize = [1,self.ksize,self.ksize,1],
			strides=[1,self.strides,self.strides,1],
			padding=self.padding,
		)

	def get_params(self):
		return []

class Flatten(Layer):

	def forward(self,X,isTrain=True):
		return tf.contrib.layers.flatten(X)

	def get_params(self):
		return []

class DenseLayer(Layer):

	def __init__(self, inp,out):
		self.W = tf.Variable((np.random.randn(inp,out) * np.sqrt(2.0 / inp)).astype(np.float32))
		self.b = tf.Variable(np.zeros(out,dtype=np.float32))
		self.params = [self.W, self.b]

	def forward(self,X,isTrain=True):
		return tf.matmul(X, self.W) + self.b

	def copyFromKerasLayers(self,layer):
		W,b = layer.get_weights()
		op1 = self.W.assign(W)
		op2= self.b.assign(b)
		#self.session.run((op1,op2))