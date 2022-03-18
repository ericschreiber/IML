#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

device = 'cpu'

#code source for k-fold cross validation: https://www.askpython.com/python/examples/k-fold-cross-validation


# In[2]:


train = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task1\task1a_do4bq81me\train.csv')

X = train.iloc[:,1:]
y = train.iloc[:,0]
print(X)
print(y)


# In[3]:


k = 10
kf = KFold(n_splits=k, random_state=None)
lambd = [0.1, 1, 10, 100, 200]

avg_acc = []


# In[4]:


for lam in lambd:
    print(lam)
    model = Ridge(alpha=lam)
    acc_score = []
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]

        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)

        acc = np.sqrt(mean_squared_error(pred_values , y_test))
        acc_score.append(acc)
        
    avg_acc_score = sum(acc_score)/k
    avg_acc.append(avg_acc_score)


# In[5]:


print('Avg accuracy : {}'.format(avg_acc))


# Submission

# In[6]:


dictionary = {'AVG': avg_acc}
avg_acc_df = pd.DataFrame(dictionary )
print(avg_acc_df)
avg_acc_df.to_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task1\submission1a.csv', index=False, header=False)


# In[ ]:




