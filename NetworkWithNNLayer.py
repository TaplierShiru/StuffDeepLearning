import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

class HiddenLayer():
  
  def __init__(self,C1,C2):
    self.C1 = C1
    self.C2 = C2
    W = np.random.randn(C1,C2) * np.sqrt(2 / C1)
    b = np.zeros(C2)
    self.W = tf.Variable(W.astype(np.float32))
    self.b = tf.Variable(b.astype(np.float32))
  
  def forward(self,X):
    return tf.nn.relu(tf.matmul(X,self.W) + self.b)

class NutsHiddenLayerNN():
  
  
  def __init__(self,K,D, layer_size, ps_keep):
    assert(len(layer_size) == len(ps_keep))
    self.ps_keep = ps_keep
    self.hidden_lay = []
    
    C1 = D
    
    for C2 in layer_size:
      temp = HiddenLayer(C1,C2)
      self.hidden_lay.append(temp)
      C1 = C2
    
    W = np.random.randn(C1,K) * np.sqrt(2 / C1)
    b = np.zeros(K)
    
    self.W = tf.Variable(W.astype(np.float32))
    self.b = tf.Variable(b.astype(np.float32))
    
    self.X = tf.placeholder(tf.float32, shape = (None, D))
    self.T = tf.placeholder(tf.float32, shape = (None, K))
    
  def set_session(self,ses):
    self.session = ses
    
  def error_rate(self,y,t):
    return np.mean(y != t)
    
  def through_network(self):
    Z = self.X
    
    for V in self.hidden_lay:
      Z = V.forward(Z)
      
    return tf.matmul(Z, self.W) + self.b
  
  def through_softmax(self,X):
    return tf.nn.softmax(self.through_network(X))
    
  
  def forward(self):
    Z = self.X
    
    for V,P in zip(self.hidden_lay, self.ps_keep):
      Z = V.forward(Z)
      Z = tf.nn.dropout(Z, P)
      
    return tf.matmul(Z, self.W) + self.b
  
  def fit(self,Xtrain,Ytrain,Xtest,Ytest,learning_rate=0.0004, mu=0.9, decay=0.999, epoch=20, batch_sz=200):    
    assert(self.session is not None)
    
    Yish = self.forward()
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=self.T))
    
    train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=mu).minimize(cost)
    
    test = self.through_network()
    
    test_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=test, labels=self.T))
    
    pred_test = tf.argmax(test, axis=1)
    
    n_batch = Xtrain.shape[0] // batch_sz
    
    init_op = tf.global_variables_initializer()
    
    cost = []
    
    self.session.run(init_op)
    
    for i in range(epoch):
      Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
      for j in range(n_batch):
        Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
        Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
        
        self.session.run(train, feed_dict = {self.X: Xbatch, self.T:Ybatch})
        
        train_costs = self.session.run(test_cost,feed_dict = {self.X:Xbatch, self.T: Ybatch})
        
        Yp = self.session.run(pred_test, feed_dict = {self.X: Xtest})
        error = self.error_rate(Yp, Ytest)
        cost.append(test_cost)
        
        if j % 30 == 0:
          print('Accuracy is ', (1-error))
          print('Cost is ', train_costs)
        
      
    plt.plot(cost)
    plt.show()