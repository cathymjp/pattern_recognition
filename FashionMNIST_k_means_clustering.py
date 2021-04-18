#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[2]:


#Download Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


X = train_images.reshape(len(train_images), -1)
Y = train_labels

#normalize the data to 0 - 1
X = X.astype(float) / 255.


# In[4]:


# Number of clusters = 10
from sklearn.cluster import KMeans
n_digits = len(np.unique(test_labels))
print(n_digits)

# Initialize K-Means model
kmeans = KMeans(n_clusters = n_digits)

# Fit the model to the trianing data
kmeans.fit(X)


# In[5]:


def infer_cluster_labels(kmeans, actual_labels):
    inferred_labels = {}

    for i in range(kmeans.n_clusters):
        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

# test the infer_cluster_labels() and infer_data_labels() functions
cluster_labels = infer_cluster_labels(kmeans, Y)
X_clusters = kmeans.predict(X)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)


# In[6]:


# Optimization and Evalution
from sklearn import metrics

def calculate_metrics(estimator, data, labels):
    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))


# In[8]:


#See how different number of clusters affect the accuracy
clusters = [10, 16, 36, 64, 144, 256]
    
# test different numbers of clusters
for n_clusters in clusters:
    estimator = KMeans(n_clusters = n_clusters)
    estimator.fit(X)
    
    calculate_metrics(estimator, X, Y)
    
    # determine predicted labels
    cluster_labels = infer_cluster_labels(estimator, Y)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)
    
    # calculate and print accuracy
    print('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y)))


# In[10]:


# test kmeans algorithm on testing dataset

# convert each image to 1 dimensional array
X_test = test_images.reshape(len(test_images),-1)

# normalize the data to 0 - 1
X_test = X_test.astype(float) / 255.

# initialize and fit KMeans algorithm on training data
# number of clusters = 10
kmeans = KMeans(n_clusters = 10)
kmeans.fit(X)
cluster_labels = infer_cluster_labels(kmeans, Y)

# predict labels for testing data
test_clusters = kmeans.predict(X_test)
predicted_labels = infer_data_labels(kmeans.predict(X_test), cluster_labels)
    
# calculate and print accuracy for k = 10
print('Accuracy: {}\n'.format(metrics.accuracy_score(test_labels, predicted_labels)))


# In[ ]:




