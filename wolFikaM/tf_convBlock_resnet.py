import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_layers import BatchNormLayer, ConvLayer, ActivationLayer

def init_filter(d,mi,mo,stride):
	return (np.random.randn(d,d,mi,mo) * np.sqrt(2.0 / (d * d * mi))).astype(np.float32)

class ConvBlock:

	def __init__(self,f_in,fm_sizes,stride=1,d=3,activation=None):
		assert(len(fm_sizes) == 3)

		if activation is None:
			self.activation = ActivationLayer()
		else:
			self.activation = activation
		#main brach
		#conv->bn->f()--->conv->bn->f()--->conv-->bn
		self.conv1 = ConvLayer(d,f_in,fm_sizes[0],stride)
		self.bn1=BatchNormLayer(fm_sizes[0])
		self.conv2 = ConvLayer(d,fm_sizes[0],fm_sizes[1],stride)
		self.bn2=BatchNormLayer(fm_sizes[1])
		self.conv3=ConvLayer(d,fm_sizes[1],fm_sizes[2],stride)
		self.bn3=BatchNormLayer(fm_sizes[2])			

		#skip
		#Conv-->BN
		self.convs=ConvLayer(d,f_in,fm_sizes[2],stride)
		self.bns=BatchNormLayer(fm_sizes[2])

		self.layers = [
			self.conv1,self.bn1, self.activation,
			self.conv2, self.bn2, self.activation,
			self.conv3, self.bn3,
			self.convs, self.bns,
		]

	def forward(self,X,isTrain=True):
		#main br
		FX = X
		for layer in self.layers[:-2]:
			FX = layer.forward(FX,isTrain)

		#skip
		SX = self.layers[-2].forward(X=X,isTrain=isTrain)
		SX = self.layers[-1].forward(X=SX,isTrain=isTrain)

		return FX + SX

	def copyFromKerasLayers(self,layers):
		self.conv1.copyFromKerasLayers(layers[0])
		self.bn1.copyFromKerasLayers(layers[1])

		self.conv2.copyFromKerasLayers(layers[3])
		self.bn2.copyFromKerasLayers(layers[4])

		self.conv3.copyFromKerasLayers(layers[6])
		self.bn3.copyFromKerasLayers(layers[8])

		self.convs.copyFromKerasLayers(layers[7])
		self.bns.copyFromKerasLayers(layers[9])

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params

