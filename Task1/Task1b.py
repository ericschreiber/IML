#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV

device = 'cpu'

#Code loosly based on https://datascience.stackexchange.com/questions/56804/sckit-learn-cross-validation-and-model-retrain
#And our exercise 1a


# In[2]:


train = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task1\task1b_ql4jfi6af0\train.csv')

X = train.iloc[:,2:]
Y = train.iloc[:,1]
ID = train.iloc[:,0]
pd.set_option("display.precision", 7)
print(X)
print(Y)
print(X.shape)
print(Y.shape)


# Wir verlieren etwas präzision durch das Laden in den DataFrame (durch float precision)

# In[3]:


def transform(X):
    l1 = X
    l2 = X*X
    l3 = np.exp(X)
    l4 = np.cos(X)
    const = np.ones((X.shape[0],1))
    return np.concatenate((l1, l2, l3, l4, const), axis=1)


# In[4]:


transformer = FunctionTransformer(transform, validate=True)


# In[5]:


#X_Test = np.array([[1, 2, 3, 4, 5], [1, 1,1, 1,1], [1, 1,1, 1,1], [1, 1,1, 1,1]])
#print(transformer.transform(X_Test))


# In[6]:


X_trans = transformer.transform(X)
print(X_trans.shape)


# In[7]:


param_grid = {'alpha' : np.logspace(-2, 20, num=100)}
kf = KFold(n_splits=10)
model = Ridge(alpha=1)

grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', return_train_score=True)


# In[8]:


best_model = grid_search.fit(X_trans, Y)
print("Lowest MSE: " + str(best_model.best_score_))
best_alpha = best_model.best_estimator_
print("With alpha: " + str(best_alpha))

coefficiants = best_model.best_estimator_.coef_
print("And coefficiants. ")
print(coefficiants)


# store
# 

# In[9]:


dictionary = {'Coefficiants': coefficiants}
avg_acc_df = pd.DataFrame(dictionary )
print(avg_acc_df)
avg_acc_df.to_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task1\submission1b.csv', index=False, header=False)


# In[ ]:




