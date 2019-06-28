import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_layers import BatchNormLayer, ConvLayer, ActivationLayer

def init_filter(d,f_in,mo,stride):
	return (np.random.randn(d,d,f_in,mo) * np.sqrt(2.0 / (d * d * f_in))).astype(np.float32)

class IdenBlock:

	def __init__(self,f_in,fout_sizes,stride=1,d=3,activation=None):
		assert(len(fout_sizes) == 2)
		assert(fout_sizes[0] == fout_sizes[1])

		if activation is None:
			self.activation = ActivationLayer()
		else:
			self.activation = activation
		#main brach
		#bn-->relu--->cov--->bn--->relu--->conv
		self.conv1 = ConvLayer(d,f_in,fout_sizes[0],stride)
		self.bn1=BatchNormLayer(fout_sizes[0])
		self.conv2 = ConvLayer(d,fout_sizes[0],fout_sizes[1],stride)
		self.bn2=BatchNormLayer(fout_sizes[1])		

		#skipS
		#just X

		self.layers = [
			self.bn1,self.activation,self.conv1,
			self.bn2,self.activation,self.conv2,
		]

	def forward(self,X,isTrain=True):
		#main br
		FX = X
		for layer in self.layers:
			FX = layer.forward(FX,isTrain)
		return FX + X

	def copyFromKerasLayers(self,layers):
	    self.conv1.copyFromKerasLayers(layers[0])
	    self.bn1.copyFromKerasLayers(layers[1])

	    self.conv2.copyFromKerasLayers(layers[3])
	    self.bn2.copyFromKerasLayers(layers[4])
	    

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
