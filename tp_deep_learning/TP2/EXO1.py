#!/usr/bin/env python
# coding: utf-8

# # Exo1: Regression Logistic avec Keras

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


# In[4]:


model = Sequential() #On créé ainsi un réseau de neurones vide                                ##
model.add(Dense(10,  input_dim=784, name='fc1')) #l’ajout d’une couche de projection linéaire ##
                                                 #(couche complètement connectée) de taille 10##
model.add(Activation('softmax')) #l’ajout d’une couche d’activation de type softmax           ##
model.summary()                                                                               ##


# # Question :
# #Quel modèle de prédiction reconnaissez-vous ? Vérifier le nombre de paramètres du réseau à apprendre dans la méthode summary(). - Écrire un script exo1.py permettant de créer le réseau de neurone ci-dessus.

# In[5]:


learning_rate = 0.1                                                                        
sgd = SGD(learning_rate) #méthode d’optimisation (descente de gradient stochastique)         
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])            

#on compile le modèle en lui passant un loss (l’ entropie croisée), une méthode d’optimisation (sgd), et une métrique d’évaluation( le taux de bonne prédiction des catégories, accuracy): #Implémenter l’apprentissage du modèle sur la base de train de la base MNIST.
#Évaluer les performances du réseau sur la base de test et les comparer à celles obtenues lors 
#de la séance précédente (ré-implémentation manuelle de l’algorithme de rétro-propagation). Conclure.
#Obtained the following results
# In[6]:


batch_size = 100 #nombre d’exemples utilisé pour estimer le gradient de la fonction de coût. ##
nb_epoch = 20 # nombre de passages sur l’ensemble des exemples de la base d’apprentissage) lors de la descente de gradient.                                               ##
K=10                       

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

#Apprentissage du modele avec la methode fit
#Le premier élément de score renvoie la fonction de coût sur la base de test, 
#le second élément renvoie le taux de bonne détection (accuracy).
scores = model.evaluate(X_test, Y_test, verbose=0) #evaluation du modele sur l'ensemble de test
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
###############################################################################################

#loss: 27.12%
#acc: 92.33%


# In[ ]:




