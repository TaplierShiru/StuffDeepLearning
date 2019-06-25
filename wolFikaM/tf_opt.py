import tensorflow as tf
import numpy as np

class Optimizer:

	def __init__(self,lr):
		self.lr= lr
		self.opt = None

	def get_optimizer(self):
		return self.opt

	def minimize(self,inp):
		return self.get_optimizer().minimize(inp)

	def init_variables(self,session):
		session.run(tf.variables_initializer(self.get_optimizer().variables()))

class MomentumOpt(Optimizer):

	def __init__(self,lr=10e-4,momentum=0.9,decay=0.99,nesterov=True):
		Optimizer.__init__(self,lr)
		self.momentum = momentum
		self.nesterov = nesterov
		self.opt = tf.train.MomentumOptimizer(learning_rate=self.lr,
			momentum=self.momentum,
			use_nesterov=self.nesterov,
		)

class RMSPropOpt(Optimizer):

	def __init__(self,lr=10e-4,decay=0.9,momentum=0.0,centered=False):
		Optimizer.__init__(self,lr)
		self.decay = decay
		self.momentum = momentum
		self.centered = centered
		self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr,
			decay=self.decay,
			momentum=self.momentum,
			centered=self.centered,
		)

class AdamOpt(Optimizer):

	def __init__(self,lr=10e-4,beta1=0.9,beta2=0.999):
		Optimizer.__init__(self,lr)
		self.beta1=beta1
		self.beta2 = beta2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.lr,
			beta1=self.beta1,
			beta2=self.beta2,
		)

class GradientDOpt(Optimizer):

	def __init__(self,lr=10e-4):
		Optimizer.__init__(self,lr)
		self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)