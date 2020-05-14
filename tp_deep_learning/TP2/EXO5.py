#!/usr/bin/env python
# coding: utf-8

# # Exercice 5 : Visualisation des représentations internes des réseaux de neurones
# 
#On va maintenant s’intéresser à visualisation de l’effet de “manifold untangling” permis par les réseaux de neurones.         
#l’objectif va être d’utiliser la méthode t-SNE de l’exercice 2 pour projeter les couches cachés des réseaux de neurones dans un espace de dimension 2, ce qui permettra de visualiser la distribution des représentations internes et des labels. 
# In[5]:


from keras.datasets import mnist
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from sklearn.mixture import *
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml


# In[2]:


def convexHulls(points, labels):
  # computing convex hulls for a set of points with asscoiated labels
  convex_hulls = []
  for i in range(10):
    convex_hulls.append(ConvexHull(points[labels==i,:]))
  return convex_hulls
def best_ellipses(points, labels):
  # computing best fiiting ellipse for a set of points with asscoiated labels
  gaussians = []
  for i in range(10):
    gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :]))
  return gaussians

def neighboring_hit(points, labels):
  k = 6
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
  distances, indices = nbrs.kneighbors(points)

  txs = 0.0
  txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  for i in range(len(points)):
    tx = 0.0
    for j in range(1,k+1):
      if (labels[indices[i,j]]== labels[i]):
        tx += 1
        tx /= k
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx

  for i in range(10):
    txsc[i] /= nppts[i]

  return txs / len(points)

def visualization(points2D, labels, convex_hulls, ellipses ,projname, nh):

  points2D_c= []
  for i in range(10):
      points2D_c.append(points2D[labels==i, :])
  # Data Visualization
  cmap =cm.tab10

  plt.figure(figsize=(3.841, 7.195), dpi=100)
  plt.set_cmap(cmap)
  plt.subplots_adjust(hspace=0.4 )
  plt.subplot(311)
  plt.scatter(points2D[:,0], points2D[:,1], c=labels,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
  plt.colorbar(ticks=range(10))

  plt.title("2D "+projname+" - NH="+str(nh*100.0))

  vals = [ i/10.0 for i in range(10)]
  sp2 = plt.subplot(312)
  for i in range(10):
      ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
      sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-',label='$%i$'%i, color=cmap(vals[i]))
  plt.colorbar(ticks=range(10))
  plt.title(projname+" Convex Hulls")

  def plot_results(X, Y_, means, covariances, index, title, color):
      splot = plt.subplot(3, 1, 3)
      for i, (mean, covar) in enumerate(zip(means, covariances)):
          v, w = linalg.eigh(covar)
          v = 2. * np.sqrt(2.) * np.sqrt(v)
          u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
          if not np.any(Y_ == i):
              continue
          plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)

          # Plot an ellipse to show the Gaussian component
          angle = np.arctan(u[1] / u[0])
          angle = 180. * angle / np.pi  # convert to degrees
          ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(0.6)
          splot.add_artist(ell)

      plt.title(title)
  plt.subplot(313)

  for i in range(10):
      plot_results(points2D[labels==i, :], ellipses[i].predict(points2D[labels==i, :]), ellipses[i].means_,
      ellipses[i].covariances_, 0,projname+" fitting ellipses", cmap(vals[i]))

  plt.savefig(projname+".png", dpi=100)
  plt.show()

def loadModel(savename):
 with open(savename+".yaml", "r") as yaml_file:
  model = model_from_yaml(yaml_file.read())
 print( "Yaml Model ",savename,".yaml loaded ")
 model.load_weights(savename+".h5")
 print( "Weights ",savename,".h5 loaded ")
 return model


# In[3]:


from keras.datasets import mnist
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


# In[6]:


# LOADING MODEL
nameModel = "my_model_MLP" #REPLACE WITH YOUR MODEL NAME
model = loadModel(nameModel)
model.summary()


# In[7]:


# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, 10)
# COMPILING MODEL
learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))


# In[8]:


#On évalue les performances du modèle chargé sur la base de test de MNIST pour vérifier son comportement.

model.pop() #permettant de supprimer la couche au sommet du modèle
#On vas l'appliquer deux fois (on supprime la couche d’activation softmax et la couche complètement connectée)

model.summary()
model.pop()
model.summary()
predict = model.predict(X_test)


# In[9]:


#Ensuite on va utiliser la méthode t-SNE mise en place à l’exercice 2 pour visualiser les représentations internes des données.

X_embedded = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(predict)
#[t-SNE] Error after 1000 iterations: 1.328868
#X_embedded_PCA = PCA(n_components=2, svd_solver='full').fit_transform(X_train_5000)
# Function Call
convex_hulls= convexHulls(X_embedded, y_test)


# In[10]:


# Function Call
ellipses = best_ellipses(X_embedded, y_test)
nh= neighboring_hit(X_embedded, y_test)


# In[11]:


visualization(X_embedded, y_test, convex_hulls, ellipses ,'t-SNE_MLP', nh)


# In[12]:


##Pour le model ConvNet my_model_ConvNet
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
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[13]:


# LOADING MODEL
nomModel = "my_model_ConvNet"
model = loadModel(nomModel)
model.summary()


# In[14]:


# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, 10)
# COMPILING MODEL
learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


# In[15]:


scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
#On évalue les performances du modèle chargé sur la base de test de MNIST pour vérifier son comportement.

model.pop() #permettant de supprimer la couche au sommet du modèle
#On vas l'appliquer deux fois (on supprime la couche d’activation softmax et la couche complètement connectée)
model.summary()
model.pop()
model.summary()
predict = model.predict(X_test)


# In[16]:


#Ensuite on va utiliser la méthode t-SNE mise en place à l’exercice 2 pour visualiser 
#les représentations internes des données.
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[17]:


X_embedded = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(predict)
# Function Call
convex_hulls= convexHulls(X_embedded, y_test)


# In[18]:


# Function Call
ellipses = best_ellipses(X_embedded, y_test)
nh= neighboring_hit(X_embedded, y_test)


# In[19]:


visualization(X_embedded, y_test, convex_hulls, ellipses ,'t-SNE_CNN', nh)

