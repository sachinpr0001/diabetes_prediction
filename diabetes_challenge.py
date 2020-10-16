#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


# In[79]:


X = pd.read_csv('training_data\\Diabetes_XTrain.csv')
Y = pd.read_csv('training_data\\Diabetes_YTrain.csv')
test = pd.read_csv('test_cases\\Diabetes_Xtest.csv')
print(X.shape)
print(Y.shape)
print(test.shape)


# In[68]:


X.head(n=5)


# In[69]:


Y.head(n=5)


# In[70]:


print(X.shape)
print(Y.shape)


# In[80]:


data_x = X.values
print(data_x.shape)
print(type(data_x))

data_y = Y.values
print(data_y.shape)
print(type(data_y))

data_test_x = test.values
print(data_test_x.shape)
print(type(data_test_x))


# In[82]:


X_train = data_x[:,0:]
print(X_train.shape)
Y_train = data_y[:,0]
print(Y_train.shape)
X_test = data_test_x
print(X_test.shape)


# In[83]:


print(type(X_train))
print(type(Y_train))
print(type(X_test))


# In[74]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

# Test Time 
def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred


# In[120]:


filename = 'output.csv'
output_list = []
for i in range(0,192):
    pred = knn(X_train, Y_train, X_test[i])
    output_list.append(int(pred))
#print(output_list)
df = pd.DataFrame(output_list)
print(df.shape)
df.to_csv('output.csv', index = False)