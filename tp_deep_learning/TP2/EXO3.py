#!/usr/bin/env python
# coding: utf-8

# # Exercice 3 : Réseaux de neurones convolutifs avec Keras
#On va maintenant étendre le perceptron de l’exercice précédent pour mettre en place un réseau de neurones convolutif profond,“Convolutionnal Neural Networks”, ConvNets.
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


plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
    plt.subplot(10,20,i+1)
    plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
    plt.axis('off')
plt.show()

#Les réseaux convolutifs manipulent des images multi-dimensionnelles en entrée (tenseurs). 
#On va donc commencer par reformater les données d’entrée afin que chaque exemple soit de taille 28×28×1
# In[4]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

model = Sequential()

conv1= Conv2D(32,kernel_size=(5, 5),activation='relu',input_shape=(28, 28, 1),padding='valid')

#32 est le nombre de filtres
#(5, 5) est la taille spatiale de chaque filtre (masque de convolution).
#padding=’valid’ correspond ignorer les bords lors du calcul (et donc à diminuer la taille 
#spatiale en sortie de la convolution).
#avec une couche de convolution la non-linéarité en sortie de la convolution,comme illustré ici dans l’exemple avec une fonction d’activation de type relu.
# In[5]:


pool1 = MaxPooling2D(pool_size=(2, 2)) #déclaration d'une couche de max-pooling

#Des couches d’agrégation spatiale (pooling), afin de permettre une invariance aux translations locales.
#(2, 2) est la taille spatiale sur laquelle l’opération d’agrégation est effectuée. 
#on obtient donc des cartes de sorties avec des tailles spatiales divisées par deux par rapport à la taille d’entrée
# In[6]:


model.add(conv1)


# In[7]:


#L'ajout d'une couche de convolution avec 32 filtres de taille 5×5, suivie d’une non linéarité de type relu 
model.add(pool1)


# In[8]:


#Ajout d’une couche de max pooling de taille 2×2.

conv2= Conv2D(16,kernel_size=(5, 5),activation='relu',input_shape=(28, 28, 1),padding='valid')
pool2 = MaxPooling2D(pool_size=(2, 2))
model.add(conv2)

#L'ajout d'une seconde couche de convolution avec 16 filtres de taille 5×5, suivie d’une non linéarité de type relu 
# In[9]:


model.add(pool2)

#puis d’une seconde couche de max pooling de taille 2×2.
# In[10]:


model.add(Flatten()) #On mait a plat les couches convolutives précédentes

#Comme dans le réseau LeNet, on considérera la sortie du second bloc convolutif comme un vecteur, ce que revient à “mettre à plat” les couches convolutives précédentes (model.add(Flatten())).
# In[11]:


model.add(Dense(100,  input_dim=784, name='fc4'))

#L'ajout d'une couche complètement connectée de taille 100,
# In[12]:


model.add(Activation('sigmoid'))

#suivie d’une non linéarité de type sigmoïde.
# In[13]:


model.add(Dense(10, name='fc5'))

#Une couche complètement connectée de taille 10,
# In[14]:


model.add(Activation('softmax'))

#suivie d’une non linéarité de type softmax.
# In[15]:


model.summary()


# In[16]:


learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


# In[17]:


from keras.callbacks import TensorBoard
batch_size = 100
nb_epoch = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
tensorboard = TensorBoard(log_dir="_mnist", write_graph=False, write_images=True)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1,callbacks=[tensorboard])


# In[18]:


scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[19]:


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
  
saveModel(model, 'my_model_ConvNet')

