# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:57:31 2019

@author: ravik
"""
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import os
import keras
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils.linear_assignment_ import linear_assignment

def calculateAccuracy(y_pred_kmeans, y_actual):
    y_actual = y.astype(np.int64)
    D = max(y_pred_kmeans.max(), y_actual.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred_kmeans.size):
        w[y_pred_kmeans[i], y_actual[i]] += 1
    ind = linear_assignment(-w)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred_kmeans.size
    return acc

cwd = os.getcwd()
 
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def calculateKMeans(x, y):
    start = time.time()
    kmeans = KMeans(n_clusters= 10 , n_init=20, n_jobs=4)
    print("Calculating KMeans.......")
    y_pred_kmeans = kmeans.fit_predict(x)
    print(y_pred_kmeans.shape)
    print("Time for claculating Kmeans: ", time.time() - start)
    return y_pred_kmeans

def printConfusionMatrix(y_pred, y):
    cm = confusion_matrix(y_pred, y)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
def printLoss(autoEncoder, epoch):
    loss = autoEncoder.history['loss']
    val_loss = autoEncoder.history['val_loss']
    epochs = range(epoch)
    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()    
    
x_train, y_train = load_mnist(cwd, kind='train')
x_test, y_test = load_mnist(cwd, kind='t10k')


x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
print(x.shape)
x = np.divide(x, 255.)

