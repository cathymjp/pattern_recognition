#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[18]:


data_train = pd.read_csv("C:\\Users\\Park\\Downloads\\Pattern_Recognition\\datasets\\fashion-mnist_train.csv")
data_test = pd.read_csv("C:\\Users\\Park\\Downloads\\Pattern_Recognition\\datasets\\fashion-mnist_test.csv")


# In[19]:


data_train_y = data_train.label
y_test = data_test.label


# In[20]:


data_train_x = data_train.drop("label", axis = 1)/256
x_test = data_test.drop("label", axis = 1)/256


# In[21]:


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


# In[22]:


#Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


# In[23]:


#Fitting Random Forest
RF_model = RandomForestClassifier().fit(x_train, y_train)


# In[24]:


#Training Accuracy
confusion_matrix(RF_model.predict(x_train),y_train)


# In[25]:


RF_model_train_acc = (RF_model.predict(x_train) == y_train).mean()
print(">> Train Accuracy =", RF_model_train_acc)


# In[26]:


#Validation Accuracy
confusion_matrix(RF_model.predict(x_valid1), y_valid1)


# In[27]:


RF_model_valid1_acc = (RF_model.predict(x_valid1) == y_valid1).mean()
print(">> Validation Accuracy =", RF_model_valid1_acc)


# In[28]:


{"TRAIN_ACC" : RF_model_train_acc, "VALID_ACC" : RF_model_valid1_acc}


# In[ ]:




