#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_features = pd.read_csv(...)
train_labels = pd.read_csv(...)
test_features = pd.read_csv(...)


# In[28]:


result = pd.DataFrame({'pid': test_features.iloc[0::12, 0].values})
testing_features = ['LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2', 'LABEL_Sepsis', 'LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']


# In[4]:


train_features = train_features.sort_values(by=['pid','Time'])
train_labels = train_labels.sort_values(by=['pid'])
test_features = test_features.sort_values(by=['pid','Time'])


# In[10]:


def make_features(inp):
    inp = inp.to_numpy()
    finished2 = np.empty((int(inp.shape[0]/12), 70))
    for index in range(int(inp.shape[0] / 12)):
        data_without = inp[index*12 : (index+1) * 12, 2:] 
        finished2[index, :35] = np.nanmean(data_without, axis=0)
        finished2[index, 35:] = np.nanvar(data_without, axis=0)
    return np.array(finished2)


# In[11]:


train_features_new = make_features(train_features)
test_feat_new = make_features(test_features)


# In[14]:


labels = train_labels
labels.pop("pid")
labels = labels.to_numpy()


# In[16]:


from sklearn.impute import SimpleImputer
imputer2 = SimpleImputer(strategy='median')
train_feat_MEDIAN = imputer2.fit_transform(train_features_new)


# In[17]:


test_feat_MEDIAN = imputer2.fit_transform(test_feat_new)


# In[18]:


train_feat_MEDIAN = np.float32(train_feat_MEDIAN)


# In[19]:


train_feat_MEDIAN = np.nan_to_num(train_feat_MEDIAN, nan=0.0, posinf=None, neginf=None)


# In[20]:


True in np.isnan(train_feat_MEDIAN)


# In[21]:


train_feat_MEDIAN.dtype


# In[22]:


True in np.isinf(train_feat_MEDIAN)


# In[23]:


test_feat_MEDIAN = np.float32(test_feat_MEDIAN)
test_feat_MEDIAN = np.nan_to_num(test_feat_MEDIAN, nan=0.0, posinf=None, neginf=None)


# Subtask 1
# ---

# In[24]:


from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(max_depth=3))])

y = labels[2000:, 0]
X = train_feat_MEDIAN[2000:]

pipe.fit(X, y)


# In[25]:


y_pred = pipe.predict_proba(train_feat_MEDIAN[:2000])[:,1]
RocCurveDisplay.from_predictions(labels[:2000, 0], y_pred)
plt.show()


# In[26]:


from sklearn.multioutput import MultiOutputClassifier
y = labels[:,:10]
model = MultiOutputClassifier(pipe).fit(train_feat_MEDIAN, y)


# In[27]:


from sklearn.metrics import roc_auc_score
y_predict = model.predict_proba(train_feat_MEDIAN)
y_predict = np.transpose(np.asarray(y_predict)[:,:,1])
roc_auc_score(y, y_predict, average=None)


# In[29]:


y_predict = model.predict_proba(test_feat_MEDIAN)
y_pred = np.transpose(np.asarray(y_predict)[:,:,1])

for index in range(10):
    result[testing_features[index]] = y_pred[:,index]


# In[30]:


result


# Subtask 2
# ---

# In[33]:


from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(max_depth=3, class_weight='balanced'))])

y = labels[2000:, 10]
X = train_feat_MEDIAN[2000:]

pipe.fit(X, y)


# In[34]:


y_pred = pipe.predict_proba(train_feat_MEDIAN[:2000])[:,1]
RocCurveDisplay.from_predictions(labels[:2000, 10], y_pred)
plt.show()


# In[35]:


y_pred = pipe.predict_proba(test_feat_MEDIAN)[:,1]

result[testing_features[10]] = y_pred


# In[36]:


result


# Subtask 3
# ---

# In[38]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#Indeces 11, 12, 13, 14
y = labels[2000:, 11]
X = train_feat_MEDIAN[2000:]

regr = RandomForestRegressor(max_depth=3, random_state=0)
regr.fit(X, y)


# In[39]:


y_prediction = regr.predict(train_feat_MEDIAN[:2000])
loss = r2_score(labels[:2000, 11], y_prediction)
loss


# In[45]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#Indeces 11, 12, 13, 14
y = labels[:,11:]
X = train_feat_MEDIAN[:]

regr = RandomForestRegressor(max_depth=3, random_state=0)
model = MultiOutputRegressor(regr).fit(X, y)


# In[46]:


y_prediction = model.predict(test_feat_MEDIAN)

for index in range(4):
    result[testing_features[index + 11]] = y_prediction[:,index]


# In[47]:


result


# In[48]:


import pandas as pd

# suppose df is a pandas dataframe containing the result
result.to_csv('prediction_Jan_Eric2.zip', index=False, float_format='%.3f', compression='zip')


# In[ ]:




