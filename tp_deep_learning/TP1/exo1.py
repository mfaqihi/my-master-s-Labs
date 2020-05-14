#!/usr/bin/env python
# coding: utf-8

# # Exercice 1 : Régression Logistique

# In[1]:


from keras.datasets import mnist
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np


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

N = X_train.shape[0]
d = X_train.shape[1]
W = np.zeros((d,K))
b = np.zeros((1,K))
numEp = 20 # Number of epochs for gradient descent
eta = 1e-1 # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))


def forward1(batch, W, b):
    numerateur = np.exp(np.dot(batch,W)+b);
    denominateur = np.sum(numerateur, axis=1)[:,None]
    y_bar = numerateur/denominateur;
    return y_bar;

def accuracy(W, b, images, labels):
  pred = forward1(images, W,b )
  return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0

for epoch in range(numEp):
  for ex in range(nb_batches):
     # FORWARD PASS : compute prediction with current params for examples in batch
     batch = X_train[ex*batch_size:(ex+1)*batch_size,:]
     y_batch = Y_train[ex*batch_size:(ex+1)*batch_size,:]
     Y_bar = forward1(batch,W,b);
     # BACKWARD PASS :
     # 1) compute gradients for W and b
     gradW = (1/batch_size)*np.dot(batch.T,(-y_batch+Y_bar))
     gradb = (1/batch_size)*np.sum((-y_batch+Y_bar),0)
     # 2) update W and b parameters with gradient descent
     W = W - eta*gradW
     b = b - eta*gradb
  print("epoch ", epoch, "accurcy train=",accuracy(W, b, X_train, Y_train), "accurcy test=",accuracy(W, b, X_test, Y_test))  

     

