#!/usr/bin/env python
# coding: utf-8

# # Exercice 2 : Perceptron multi-couches (MLP)

# In[1]:


import numpy as np
import time

from keras.datasets import mnist
from keras.utils import np_utils


# In[2]:


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[3]:


K=10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)
d = X_train.shape[1]
N = X_train.shape[0]
L = 100
sigma = 0.1
Wh = np.random.randn(d,L) * sigma 
bh = np.zeros((1,L))
Wy = np.random.randn(L,K) * sigma 
by = np.zeros((1,K))

numEp = 100 # Number of epochs for gradient descent
eta = 1.0 # Learning rate


def softmax(X):
 # Input matrix X of size Nbxd - Output matrix of same size
 E = np.exp(X)
 return (E.T / np.sum(E,axis=1)).T

def forward2(batch, Wh, bh, Wy, by):
    #y_bar = np.dot(X_train,W);
    u = np.dot(batch,Wh)+bh
    h= 1/(1+np.exp(-1*u));
    v = np.dot(h,Wy)+by
    y_bar = softmax(v)
    return y_bar, h;

def accuracy2(Wh, bh, Wy, by, images, labels):
  pred, Hn = forward2(images, Wh, bh, Wy, by )
  return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0


batch_size = 100
nb_batches = int(float(N) / batch_size)

for epoch in range(numEp):
  for ex in range(nb_batches):
     # FORWARD PASS : compute prediction with current params for examples in batch
     batch = X_train[ex*batch_size:(ex+1)*batch_size,:]
     y_batch = Y_train[ex*batch_size:(ex+1)*batch_size,:]
     Y_bar, H = forward2(batch, Wh, bh, Wy, by);
     Y_bar, H = forward2(batch, Wh, bh, Wy, by)
     # BACKWARD PASS :
     # 1) compute gradients for W and b
     
     gradv = Y_bar - y_batch
        # gradWy = h^T * deltay - (LxN) * (NxK)
     gradWy = 1.0/batch_size * np.matmul(np.transpose(H),gradv)
     gradby = 1.0/batch_size * (gradv.sum(axis=0)).reshape((1,K))
        # deltah   =dE / dh~ size (N,L)
     deltah = np.matmul(gradv, np.transpose(Wy)) * (H* (1.0-H))
    # gradWh = x^T * deltah - (dxN) * (NxL) = (dxL)
     gradWh = 1.0/batch_size * np.matmul(np.transpose(X_train[ex*batch_size:(ex+1)*batch_size,:]),deltah)
     gradbh = 1.0/batch_size * (deltah.sum(axis=0)).reshape((1,L)) 
     Wy = Wy - eta * gradWy
     by = by - eta * gradby
        
     Wh = Wh - eta * gradWh
     bh = bh - eta * gradbh
    


  print("epoch ", epoch, "accuracy train=",accuracy2(Wh, bh, Wy, by, X_train, Y_train), "accuracy test=",accuracy2(Wh, bh, Wy, by, X_test, Y_test))

