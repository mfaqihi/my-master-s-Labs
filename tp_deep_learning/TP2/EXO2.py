#!/usr/bin/env python
# coding: utf-8

# # Exercice 2 : Perceptron avec Keras

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import mnist


# In[2]:


from keras.models import model_from_yaml
def saveModel(model, savename):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename+".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    print("Yaml Model ",savename,".yaml saved to disk")
  # serialize weights to HDF5
  model.save_weights(savename+".h5")
  print("Weights ",savename,".h5 saved to disk")


# In[3]:


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


# In[4]:


plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
    plt.subplot(10,20,i+1)
    plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
    plt.axis('off')
plt.show()

#On va maintenant enrichir le modèle de régression logistique en créant une couche de neurones cachée complètement connectée supplémentaire, suivie d’une fonction d’activation non linéaire de type sigmoïde. 
#On va ainsi obtenir un réseau de neurones à une couche cachée, le Perceptron (cf séance précédente)
# In[5]:


model = Sequential()
model.add(Dense(100,  input_dim=784, name='fc2'))
model.add(Activation('sigmoid'))
model.add(Dense(10, name='fc3'))
model.add(Activation('softmax'))
model.summary()


# In[6]:


learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


# In[7]:


batch_size = 100
nb_epoch = 100
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)


# In[8]:


scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[9]:


saveModel(model, 'my_model_MLP')

