#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


# Import Fashion MNIST using keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


# 3-layer Neural Network ClassifierBuild the model
# 1) Flatten image to 28 x 28 = 784 vector
# 2) 128 neurons and relu function
# 3) 10 neurons and softmax function

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])


# In[4]:


# Compile the Model using Adam optimizer
model.compile(
   optimizer = tf.optimizers.Adam(), 
   loss = 'sparse_categorical_crossentropy', 
   metrics = ['accuracy'])


# In[5]:


# Train the model
model.fit(train_images, train_labels, epochs = 15, batch_size = 32)


# In[7]:


# Evaluate the model
model.evaluate(test_images, test_labels)


# In[ ]:




