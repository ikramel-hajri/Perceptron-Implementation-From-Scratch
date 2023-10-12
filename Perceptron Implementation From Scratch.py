#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Lab 1
# ## Perceptron Implementation From Scratch
# 

# 1 Environment Installation
# During the deep learning labs, we will use Python (3.7 or later) as a programming language as well
# as the following Python packages: numpy, scipy, pandas, scikit-learn, scikit-image, matplotlib,
# tensorflow and jupyterlab. To set up the environment, either install “Anaconda” 1 or manually
# install the packages above using “pip” for a minimal footprint environment. If you succeeded in
# installing the environment, then go to Section 2.

# In[2]:


pip install numpy scipy pandas scikit-learn scikit-image matplotlib tensorflow


# ## 2.1 Data Loading and Preprocessing

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# In[2]:


# Step 1: Load the data using pandas
df = pd.read_csv('disease.csv')
df.sample(10)


# In[3]:


# Step 2: Split the data into training and testing sets

X, y = df.iloc[:,:-1].values , df.iloc[:,-1].values
#X = data.drop('disease_presence', axis=1) 
#y = data['disease_presence']

X.shape , y.shape


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle = True, random_state=13)
#we should alwyas check the shape(the number of rows and columns of a given DataFrame)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[5]:


# Step 3: Extract a validation set from the training set
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
X_train_.shape, X_val.shape, y_train_.shape, y_val.shape


# fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal object's state. Afterwards, you can call its transform() method to apply the transformation to any particular set of examples.
# 
# fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x
# , while also returning the transformed x′
# . Internally, the transformer object just calls first fit() and then transform() on the same data.

# In[6]:


# Step 4: Scale the features
scaler = StandardScaler()
scaler.fit(X_train_)
X_train_scaled = scaler.transform(X_train_)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# we dont fit the X_val because we dont have to touch ut once we  will deploy our model and we'll get a new unseen data we have to fit but normally we cant that's why we keep it untouched

# 1. Perceptron model: define a function taking as input a 2-dimensional numpy array representing the features (X), the Perceptron’s weights (W) and bias (b), and returning a prediction
# vector (y). The activation function to use is the identity (g(x) = x).

# In[7]:


def percepton(X,W,b):
    
    y = np.dot(X,W)+b
    return y.flatten()
# no need to add an identity


# Define a function that computes the mean squared error (MSE), taking as inputs the observed values (y_true) and the predicted values (y_pred).

# In[31]:


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 3. Define a function that computes the gradients (of the weights and the bias).

# In[9]:


def gradient_b(y_pred, y_true):
    N= y_pred.shape[0] # y_pred.len()
    return (2./N) * np.sum(y_pred - y_true) *1 # w0 =1 derive(w0) =1

def gradient_W(y_pred, y_true, X):
    N= y_pred.shape[0] # y_pred.len()
    y_pred = y_pred.reshape
    Y = (y_pred - y_true).reshape((N,1 )) #to make sure it's one column and N lines
    return (2. / N)* np.dot(np.transpose(X), Y)


# Initialize randomly (using the normal distribution) the weights and the bias of the perceptron.

# In[10]:


get_ipython().run_line_magic('pinfo', 'np.random')


# In[11]:


b= np.random.normal(0,1)
W = np.random.normal(0,1 ,size = (X_train_scaled.shape[1],1))


# Compute the predictions on the training set.

# In[12]:


y_pred = percepton(X_train_scaled,W,b)


# 6. Compute the gradients..

# In[19]:


grad_b = gradient_b(y_pred, y_train_)
grad_W = gradient_W(y_pred, y_train_ , X_train_scaled)


# 7. Update the Perceptron’s parameters using the gradient descent update rule.
# Use a learning rate of 0.001.

# In[22]:


lr = 1e-4 # 0.001
b = b - lr * grad_b
W = W - lr * grad_W


# 8. Compute the MSE loss

# In[32]:


loss_MSE = MSE(y_train_, y_pred)


# 9. Repeat these steps for 10,000 epochs.

# In[40]:


# initialize W and b
b = np.random.normal(0,1)
W = np.random.normal(0,1, size=(X_train_scaled.shape[1],1))
# initialize the learning rate
train_losses = list()
lr = 1e-4
# training
for i in range(1000):
    y_pred = percepton(X_train_scaled,W,b)
    grad_b = gradient_b(y_pred, y_train_)
    grad_W = gradient_W(y_pred, y_train_ , X_train_scaled)
    # update parameters
    b = b - lr * grad_b
    W = W - lr * grad_W
    # compute MSE loss
    loss_train = MSE(y_train_, y_pred)
    train_losses.append(loss_train)
    print(f'Epoch {i}, MSE ={loss_train}')


# In[ ]:




