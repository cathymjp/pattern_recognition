#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


#Download Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[4]:


data_train = pd.read_csv("C:\\Users\\Park\\Downloads\\Pattern_Recognition\\datasets\\fashion-mnist_train.csv")
data_test = pd.read_csv("C:\\Users\\Park\\Downloads\\Pattern_Recognition\\datasets\\fashion-mnist_test.csv")


# In[5]:


data_train_y = data_train.label
y_test = data_test.label


# In[6]:


data_train_x = data_train.drop("label", axis = 1)/256
x_test = data_test.drop("label", axis = 1)/256


# In[7]:


np.random.seed(0)
valid2_idx = np.random.choice(60000,10000,replace = False)
valid1_idx = np.random.choice(list(set(range(60000)) - set(valid2_idx)), 10000,replace = False)
train_idx = list(set(range(60000)) - set(valid1_idx) - set(valid2_idx))

x_train = data_train_x.iloc[train_idx,:]
y_train = data_train_y.iloc[train_idx]

x_valid1 = data_train_x.iloc[valid1_idx,:]
y_valid1 = data_train_y.iloc[valid1_idx]

x_valid2 = data_train_x.iloc[valid2_idx,:]
y_valid2 = data_train_y.iloc[valid2_idx]


# In[8]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


# In[9]:


#Bayesian Classifier
BC_model = GaussianNB().fit(x_train, y_train)


# In[10]:


confusion_matrix(BC_model.predict(x_train), y_train)


# In[11]:


BC_model_train_acc = (BC_model.predict(x_train) == y_train).mean()
print(">> Training Accuracy =", BC_model_train_acc)


# In[12]:


#Validation Accuracy
confusion_matrix(BC_model.predict(x_valid1), y_valid1)


# In[13]:


BC_model_valid1_acc = (BC_model.predict(x_valid1) == y_valid1).mean()
print(">> Validation Accuracy =", BC_model_valid1_acc)


# In[15]:


{"TRAIN_ACC" : BC_model_train_acc, "VALID_ACC" : BC_model_valid1_acc}


# In[ ]:




