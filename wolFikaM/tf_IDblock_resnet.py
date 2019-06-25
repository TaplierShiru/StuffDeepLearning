import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_layers import BatchNormLayer, ConvLayer, ActivationLayer

def init_filter(d,f_in,mo,stride):
	return (np.random.randn(d,d,f_in,mo) * np.sqrt(2.0 / (d * d * f_in))).astype(np.float32)

class IdenBlock:

	def __init__(self,f_in,fout_sizes,stride=1,d=3,activation=None):
		assert(len(fout_sizes) == 3)
		assert(fout_sizes[0] == fout_sizes[2])

		if activation is None:
			self.activation = ActivationLayer()
		else:
			self.activation = activation
		#main brach
		#conv->bn->f()--->conv->bn->f()--->conv-->bn
		self.conv1 = ConvLayer(d,f_in,fout_sizes[0],stride)
		self.bn1=BatchNormLayer(fout_sizes[0])
		self.conv2 = ConvLayer(d,fout_sizes[0],fout_sizes[1],stride,'SAME')
		self.bn2=BatchNormLayer(fout_sizes[1])
		self.conv3=ConvLayer(d,fout_sizes[1],fout_sizes[2],stride)
		self.bn3=BatchNormLayer(fout_sizes[2])			

		#skipS
		#just X

		self.layers = [
			self.conv1,self.bn1, self.activation,
			self.conv2, self.bn2, self.activation,
			self.conv3, self.bn3,
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
	    
	    self.conv3.copyFromKerasLayers(layers[6])
	    self.bn3.copyFromKerasLayers(layers[7])

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
