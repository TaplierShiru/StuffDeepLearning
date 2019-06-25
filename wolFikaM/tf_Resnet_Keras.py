import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tf_layers import ConvLayer, BatchNormLayer, ReLULayer, MaxPoolLayer
from tf_IDblock_resnet import IdenBlock
from tf_convBlock_resnet import ConvBlock

class AvgPool:

	def __init__(self, ksize):
		self.ksize = ksize

	def forward(self,X):
		return tf.nn.avg_pool(
			X,
			ksize = [1,self.ksize,self.ksize,1],
			strides=[1,1,1,1],
			padding='VALID',
		)

	def get_params(self):
		return []

class Flatten:

	def forward(self,X):
		return tf.contrib.layers.flatten(X)

	def get_params(self):
		return []

def custom_softmax(x):
	m = tf.reduce_max(x,1)
	x = x - m
	e = tf.exp(x)
	return e / tf.reduce_sum(e,-1)

class DenseLayer:

	def __init__(self, mi,mo):
		self.W = tf.Variable((np.random.randn(mi,mo) * np.sqrt(2.0 / mi)).astype(np.float32))
		self.b = tf.Variable(np.zeros(mo,dtype=np.float32))

	def forward(self,X):
		return tf.matmul(X, self.W) + self.b

	def copyFromKerasLayers(self,layer):
		W,b = layer.get_weights()
		op1 = self.W.assign(W)
		op2= self.b.assign(b)
		self.session.run((op1,op2))

	def get_params(self):
		return [self.W, self.b]

class TFResNet:

	def __init__(self):

		self.layers = [
		  # before conv block
		  
		  ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
		  BatchNormLayer(64),
		  ReLULayer(),
		  MaxPoolLayer(dim=3),
		  # conv block
		  ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
		  # identity block x 2
		  IdenBlock(mi=256, fm_sizes=[64, 64, 256]),
		  IdenBlock(mi=256, fm_sizes=[64, 64, 256]),
		  # conv block
		  ConvBlock(mi=256, fm_sizes=[128, 128, 512], stride=2),
		  # identity block x 3
		  IdenBlock(mi=512, fm_sizes=[128, 128, 512]),
		  IdenBlock(mi=512, fm_sizes=[128, 128, 512]),
		  IdenBlock(mi=512, fm_sizes=[128, 128, 512]),
		  # conv block
		  ConvBlock(mi=512, fm_sizes=[256, 256, 1024], stride=2),
		  # identity block x 5
		  IdenBlock(mi=1024, fm_sizes=[256, 256, 1024]),
		  IdenBlock(mi=1024, fm_sizes=[256, 256, 1024]),
		  IdenBlock(mi=1024, fm_sizes=[256, 256, 1024]),
		  IdenBlock(mi=1024, fm_sizes=[256, 256, 1024]),
		  IdenBlock(mi=1024, fm_sizes=[256, 256, 1024]),
		  # conv block
		  ConvBlock(mi=1024, fm_sizes=[512, 512, 2048], stride=2),
		  # identity block x 2
		  IdenBlock(mi=2048, fm_sizes=[512, 512, 2048]),
		  IdenBlock(mi=2048, fm_sizes=[512, 512, 2048]),
		  # pool / flatten / dense
		  AvgPool(ksize=7),
		  Flatten(),
		  DenseLayer(mi=2048, mo=1000),
		]

		self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
		self.output = self.forward(self.input_)

	def copyFromKerasLayers(self, layers):
		print(layers[173:])
		# conv
		self.layers[0].copyFromKerasLayers(layers[2])
		# bn
		self.layers[1].copyFromKerasLayers(layers[3])
		# cb
		self.layers[4].copyFromKerasLayers(layers[7:19]) # size=12
		# ib x 2
		self.layers[5].copyFromKerasLayers(layers[19:29]) # size=10
		self.layers[6].copyFromKerasLayers(layers[29:39])
		# cb
		self.layers[7].copyFromKerasLayers(layers[39:51])
		# ib x 3
		self.layers[8].copyFromKerasLayers(layers[51:61])
		self.layers[9].copyFromKerasLayers(layers[61:71])
		self.layers[10].copyFromKerasLayers(layers[71:81])
		# cb
		self.layers[11].copyFromKerasLayers(layers[81:93])
		# ib x 5
		self.layers[12].copyFromKerasLayers(layers[93:103])
		self.layers[13].copyFromKerasLayers(layers[103:113])
		self.layers[14].copyFromKerasLayers(layers[113:123])
		self.layers[15].copyFromKerasLayers(layers[123:133])
		self.layers[16].copyFromKerasLayers(layers[133:143])
		# cb
		self.layers[17].copyFromKerasLayers(layers[143:155])
		# ib x 2
		self.layers[18].copyFromKerasLayers(layers[155:165])
		self.layers[19].copyFromKerasLayers(layers[165:175])
		# dense
		self.layers[22].copyFromKerasLayers(layers[176])

	def forward(self,X):
		for layer in self.layers:
			X = layer.forward(X)
		return X

	def predict(self,X):
		assert(self.session is not None)
		return self.session.run(
			self.output,
			feed_dict={self.input_:X},
		)

	def set_session(self,session):
		self.session= session
		for layer in self.layers:
			if isinstance(layer,ConvBlock) or isinstance(layer,IdenBlock):
				layer.set_session(session)
			else:
				layer.session = session

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params

if __name__ == '__main__':
	resnet_ = ResNet50(weights='imagenet')

	x = resnet_.layers[-2].output
	W,b = resnet_.layers[-1].get_weights()
	y = Dense(1000)(x)
	resnet = Model(resnet_.input, y)
	resnet.layers[-1].set_weights([W,b])


	partial_model = Model(
		inputs = resnet.input,
		outputs=resnet.layers[176].output,
	)

	print(partial_model.summary())
	 # create an instance of our own model
	my_partial_resnet = TFResNet()

	# make a fake image
	X = np.random.random((1, 224, 224, 3))

	# get keras output
	keras_output = partial_model.predict(X)

	### get my model output ###

	# init only the variables in our net
	init = tf.variables_initializer(my_partial_resnet.get_params())

	# note: starting a new session messes up the Keras model
	session = keras.backend.get_session()
	my_partial_resnet.set_session(session)
	session.run(init)

	# first, just make sure we can get any output
	first_output = my_partial_resnet.predict(X)
	print("first_output.shape:", first_output.shape)

	# copy params from Keras model
	my_partial_resnet.copyFromKerasLayers(partial_model.layers)
	print(f'answer {np.argmax(first_output)}')
	# compare the 2 models
	output = my_partial_resnet.predict(X)
	print(f'keras is {np.argmax(output)}')
	diff = np.abs(output - keras_output).sum()
	if diff < 1e-10:
		print("Everything's great!")
	else:
		print("diff = %s" % diff)