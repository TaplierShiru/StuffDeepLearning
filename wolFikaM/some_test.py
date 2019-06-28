from tf_layers import ActivLeakyRelu, BatchNormLayer, Flatten, DenseLayer, MaxPoolLayer, ConvLayer,ActivationLayer
from tf_convBlock_resnet import ConvBlock
from tf_IDblock_resnet import IdenBlock
from tf_opt import AdamOpt
import tensorflow as tf
from CNN import Model

import numpy as np
from tensorflow.keras.datasets import mnist

layers = [
      ConvBlock(f_in=1,fm_sizes=[32,32,64]),
      BatchNormLayer(64),
      MaxPoolLayer(),
      #14
      IdenBlock(64,[64,64]),
      IdenBlock(64,[64,64]),
      MaxPoolLayer(),
      #7
      IdenBlock(64,[64,64]),
      ConvBlock(f_in=64,fm_sizes=[64,128,256]),
      ConvLayer(d=3,f_in=256,f_out=512,padding='VALID'),
      
      Flatten(),

      DenseLayer(12800,512),
      BatchNormLayer(512),
      ActivationLayer(),
      DenseLayer(512,10),
]

ConvModel = Model(layers,input_shape=[48,28,28,1],num_classes=10)

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
    
Xtrain = Xtrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)

Xtrain /= 255
Xtest /= 255

Ytrain = Ytrain.reshape(len(Ytrain),)
Ytest = Ytest.reshape(len(Ytest),)

Xtrain = Xtrain.reshape((Xtrain.shape[0],28,28,1))
Xtest = Xtest.reshape((Xtest.shape[0],28,28,1))


session = tf.Session()
ConvModel.set_session(session)

epoch = 5
lr=10e-4
opt = AdamOpt(lr=lr)

one = ConvModel.fit_pure(Xtrain,Ytrain,Xtest,Ytest,optimizer=opt,epoch=epoch,show_figure=True)