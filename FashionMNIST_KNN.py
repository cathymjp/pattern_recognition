#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


data_train = pd.read_csv("C:\\Users\\Park\\Downloads\\Pattern_Recognition\\datasets\\fashion-mnist_train.csv")
data_test = pd.read_csv("C:\\Users\\Park\\Downloads\\Pattern_Recognition\\datasets\\fashion-mnist_test.csv")


# In[3]:


data_train_y = data_train.label
y_test = data_test.label


# In[4]:


data_train_x = data_train.drop("label", axis = 1)/256
x_test = data_test.drop("label", axis = 1)/256


# In[5]:


#Spliting and valid and training
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


# In[6]:


# K-Nearest Neighbors for k = 1
# Import K Nearest Neighbors Classifers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


# In[7]:


# k = 1
KNN_model_type1 = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)


# In[10]:


# Training
confusion_matrix(KNN_model_type1.predict(x_train), y_train)


# In[11]:


# Calculate the training accuracy for KNN when k = 1
KNN_model_type_1_train_acc = (KNN_model_type1.predict(x_train) == y_train).mean()
print(">> Training Accuracy =", KNN_model_type_1_train_acc)


# In[12]:


# Validation Accuracy
confusion_matrix(KNN_model_type1.predict(x_valid1), y_valid1)


# In[13]:


# Print validation accuracy for KNN when k = 1
KNN_model_type_1_valid1_acc = (KNN_model_type1.predict(x_valid1) == y_valid1).mean()
print(">> Validation Accuracy =", KNN_model_type_1_valid1_acc)


# In[14]:


{"TRAIN_ACC" : KNN_model_type_1_train_acc, "VALID_ACC" : KNN_model_type_1_valid1_acc}


# In[15]:


# KNN for k = 3
KNN_model_type_3 = KNeighborsClassifier(n_neighbors = 3).fit(x_train, y_train)


# In[16]:


# Training Accuracy
confusion_matrix(KNN_model_type_3.predict(x_train), y_train)


# In[18]:


# Print training accuracy for KNN when k = 3
KNN_model_type_3_train_acc = (KNN_model_type_3.predict(x_train) == y_train).mean()
print(">> Training Accuracy =", KNN_model_type_3_train_acc)


# In[19]:


#Validation Accuracy
confusion_matrix(KNN_model_type_3.predict(x_valid1), y_valid1)


# In[20]:


KNN_model_type3_valid1_acc = (KNN_model_type_3.predict(x_valid1) == y_valid1).mean()
print(">> Validation Accuracy =", KNN_model_type3_valid1_acc)


# In[22]:


{"TRAIN_ACC" : KNN_model_type_3_train_acc, "VALID_ACC" : KNN_model_type3_valid1_acc}


# In[23]:


#KNN with k = 5
KNN_model_type_5 = KNeighborsClassifier(n_neighbors = 5).fit(x_train, y_train)


# In[24]:


# Training Accuracy with k = 5
confusion_matrix(KNN_model_type_5.predict(x_train), y_train)


# In[25]:


KNN_model_type_5_train_acc = (KNN_model_type_5.predict(x_train) == y_train).mean()
print(">> Training Accuracy = ", KNN_model_type_5_train_acc)


# In[26]:


confusion_matrix(KNN_model_type_5.predict(x_valid1), y_valid1)


# In[27]:


KNN_model_type_5_valid1_acc = (KNN_model_type_5.predict(x_valid1) == y_valid1).mean()
print(">> Validation Accuracy =", KNN_model_type_5_valid1_acc)


# In[28]:


{"TRAIN_ACC" : KNN_model_type_5_train_acc, "VALID_ACC" : KNN_model_type_5_valid1_acc}


# In[ ]:




