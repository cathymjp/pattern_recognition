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


#Gradient Boosting
#Import Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


# In[7]:


#Fitting Gradient Boosting
GBST_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5).fit(x_train, y_train)


# In[9]:


#Training Accuracy
confusion_matrix(GBST_model.predict(x_train), y_train)


# In[11]:


GBST_model_train_acc = (GBST_model.predict(x_train) == y_train).mean()
print(">> Training Accuracy =", GBST_model_train_acc)


# In[12]:


#Validation Accuracy
confusion_matrix(GBST_model.predict(x_valid1), y_valid1)


# In[13]:


# Print Validation Accuracy
GBST_model_valid1_acc = (GBST_model.predict(x_valid1) == y_valid1).mean()
print(">> Validation Accuracy =", GBST_model_valid1_acc)


# In[14]:


{"TRAIN_ACC" : GBST_model_train_acc, "VALID_ACC" : GBST_model_valid1_acc}


# In[ ]:




