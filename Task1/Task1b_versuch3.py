#!/usr/bin/env python
# coding: utf-8

# Versuch mit Lasso regression

# In[6]:


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression


device = 'cpu'


# In[7]:


train = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task1\task1b_ql4jfi6af0\train.csv')

X = train.iloc[:,2:]
y = train.iloc[:,1]
ID = train.iloc[:,0]
pd.set_option("display.precision", 7)
#print(X)
#print(y)
print(X.shape)
print(y.shape)


# In[8]:


def transform(X):
    l1 = X
    l2 = X*X
    l3 = np.exp(X)
    l4 = np.cos(X)
    const = np.ones((X.shape[0],1))
    return np.concatenate((l1, l2, l3, l4, const), axis=1)


# In[9]:


transformer = FunctionTransformer(transform, validate=True)
X_trans = transformer.transform(X)
print(X_trans.shape)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split( X_trans, Y, test_size=0.15, random_state=42)


# In[ ]:





# In[19]:


reg = LassoCV(cv=5, random_state=10).fit(X_train, y_train)
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))


# In[25]:


pred_values = reg.predict(X_test)
overfitPar = reg.predict(X_train)
acc = np.sqrt(mean_squared_error(pred_values , y_test))
accOverfit = np.sqrt(mean_squared_error(overfitPar , y_train))
print("RMSE test= "+ str(acc))
print("RMSE overfit= "+ str(accOverfit))


# In[26]:


reg.coef_


# In[27]:


reg.mse_path_ #Mean square error for the test set on each fold, varying alpha.


# In[ ]:




