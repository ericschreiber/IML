#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold 
#from sklearn.linear_model import Ridge
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import FunctionTransformer
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LassoCV
#from sklearn.datasets import make_regression
from sklearn import svm
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



import seaborn as sb

device = 'cpu'


# In[60]:


train_feat = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task2\task2_k49am2lqi\train_features.csv')
train_label = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task2\task2_k49am2lqi\train_labels.csv')
test_feat = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task2\task2_k49am2lqi\test_features.csv')


# In[61]:


test_feat


# In[62]:


Abgabe = pd.DataFrame({'pid': test_feat.iloc[0::12, 0].values})
Abgabe


# In[63]:


train_feat = train_feat.sort_values(by=['pid','Time'])
#test_feat = test_feat.sort_values(by=['pid','Time'])
train_label = train_label.sort_values(by=['pid'])


# In[64]:


train_feat


# In[65]:


train_label


# In[66]:


train_feat_np = train_feat.to_numpy()
siz = int(len(train_feat_np)/12)
#print(siz)

X = np.zeros((siz,37*12))
for pid in range(siz):
    for feature in range(37):
        for time in range(12):
            X[pid,feature*12+time] = train_feat_np[pid*12+time,feature]


# In[67]:


tmp = [[val+"_"+str(i) for i in range(12)] for val in train_feat.columns]
keys = []
for sublist in tmp:
    for item in sublist:
        keys.append(item)

X_df = pd.DataFrame(columns=keys, index=range(18995), data = X)

for i in range(1,12):
    X_df.pop("Age_"+str(i))
for i in range(0,12):
    X_df.pop("pid_"+str(i))
    X_df.pop("Time_"+str(i))


# In[68]:


X_df


# In[69]:


#print(train_feat)
#train_feat_regroup = train_feat.groupby('pid').agg(lambda x: x.tolist())
#train_feat_regroup
#train_feat_np = train_feat.to_numpy()
#print(train_feat_np)


# In[70]:


#train_feat_regroup.shape
#train_feat.groupby('pid').agg(lambda x: print(len(x)) if len(x)==12 else print("problem!!")) #check if all have 12 timestamps or if timestamps are missing


# In[71]:


#train_feat_ID = train_feat.iloc[:,0]
#print(train_feat_ID)
#pd.set_option("display.precision", 7)
#print(X)
#print(y)
#print(X.shape)
#print(train_feat.shape)
#print(train_label.shape)


# Labels
# 

# In[72]:


labels = train_label
labels.pop("pid")


# In[73]:


labels


# In[74]:


labels = labels.to_numpy()


# New Version of Datahandling

# In[75]:


def make_features(data):
    a = []
    calc_feat = [np.nansum,  np.nanmean, np.nanvar,np.nanmedian, np.nanmax, np.nanmin]
    
    for i in range(int(data.shape[0] / 12)):
        data_without = data[i*12 : (i+1) * 12, 2:] #everything after Age 
        features = np.empty((6, data[:, 2:].shape[1]))
        
        for i, feat in enumerate(calc_feat):
            features[i] = feat(data_without, axis=0)
        a.append(features)
    return np.asarray(a)


# In[76]:


train_feat_new = make_features(train_feat.to_numpy())
test_feat_new = make_features(test_feat.to_numpy())


# In[77]:


train_feat_new = np.resize(train_feat_new, ((18995, 210)))
train_feat_new.shape


# In[78]:


test_feat_new = np.resize(test_feat_new, ((12664, 210)))
print(test_feat_new.shape)


# In[48]:


np.isnan(train_feat_new)


# In[57]:


imputer = KNNImputer(n_neighbors=3, weights='distance')
train_feat_new2 = imputer.fit_transform(train_feat_new)


# In[58]:


np.save("X_imputed_3_distance", train_feat_new2)


# In[79]:


train_feat_new2 = np.load("X_imputed_3_distance.npy")


# In[80]:


imputer = KNNImputer(n_neighbors=3, weights='distance')
test_feat_new2 = imputer.fit_transform(test_feat_new)
np.save("X_test_imputed_3_distance.npy", test_feat_new2)


# In[ ]:





# In[148]:


from sklearn.impute import SimpleImputer
imputer2 = SimpleImputer(strategy='median')
train_feat_new2MEDIAN = imputer2.fit_transform(train_feat_new)


# In[ ]:





# Check if we can drop some variables
# ---
# 

# In[50]:


Train_feat_corr = train_feat.corr()
Train_label_corr = train_label.corr()
#print(Train_corr)


# In[51]:


sb.set(rc={"figure.figsize":(40, 40)}) #width=8, height=4
dataplot = sb.heatmap(Train_feat_corr, cmap="YlGnBu", annot=True)
#plt.show()


# In[52]:


#dataplot = sb.heatmap(Train_label_corr, cmap="YlGnBu", annot=True)


# We should not drop any (perhaps the ones with 0.96 and higher)

# Imputing
# ---

# In[15]:


X = X_df.to_numpy()
imputer = KNNImputer(n_neighbors=5, weights='distance')
X = imputer.fit_transform(X)


# In[ ]:


np.save("X_imputed_5_distance", X)


# In[16]:


X = np.load("X_imputed_5_distance.npy")


# Select best
# 

# In[17]:


X


# In[18]:


y = labels[:, 0] #try to predict one label


# In[19]:


X_new = SelectKBest(f_classif, k=20).fit_transform(X, y)
X_new


# Subtask 1
# ---
# 
# try a linear network. Falls wir alle inputdimensionen gleichzeitig reingeben müssen ist eine svm besser, da wir dann nicht genügend Daten haben für ein lin netzwerk ~13Mio parameter für 2 layer FC -> weitere Idee sind LSTM nutzen (aber kompliziert)

# In[185]:


class CustomDataset(Dataset):
    def __init__(self, X, labels, batch_size=64, transform=None, target_transform=None):
        self.labels = labels
        self.X = X
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.labels[idx, 0]
        if self.transform:
            image = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


# In[186]:


class Linear2(nn.Module):
    def __init__(self, in_Dim, out_Dim):
        super(Linear2, self).__init__()
 
        self.lin0 = nn.Linear(in_features=in_Dim, out_features=512)
        self.lin1 = nn.Linear(in_features=512, out_features=out_Dim)
        
 
    def forward(self, x):
        x = self.lin0(x)
        x = F.relu(x)
        prediction = torch.sigmoid(self.lin1(x))
        return prediction


# Learning parameters

# In[187]:


epochs = 10
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[188]:


model = Linear2(X.shape[1], 1 ).to(device) #labels.shape[1] ).to(device)
print(model)


# In[230]:


opt = optim.Adam(model.parameters(), lr=lr)
loss = nn.BCELoss()


# In[232]:


train_data = X[:-2000]
val_data = X[-2000:]
print(len(train_data))

train_data = train_data.astype(dtype=np.float32)
val_data = val_data.astype(dtype=np.float32)
y = y.astype(dtype=np.float32)
print(train_data.dtype)
print(val_data.dtype)


# In[191]:


train_loader = CustomDataset(
    train_data,
    y,
    batch_size=batch_size,
)
val_loader = CustomDataset(
    val_data,
    y,
    batch_size=batch_size,
)


# In[239]:


def fit(model, X, labels, loss):
    X.to(device)
    labels.to(device)
    model.train
    running_loss = 0.0
   
    #for i in range(len(labels)):
    print(X.size())
    opt.zero_grad()

    inp = model(X)        

    output = loss(inp, labels)


    running_loss += output.item()
    output.backward()

    opt.step()
        
    train_loss = running_loss
    return train_loss


# In[240]:


def validate(model, X, labels, loss):
    X.to(device)
    labels.to(device)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        #for i in range(len(labels)):
       
        opt.zero_grad()

        inp = model(X)        

        output = loss(inp, labels)



        running_loss += output.item()
            
    val_loss = running_loss
    return val_loss


# In[241]:


train_loss = []
val_loss = []
torch.backends.cudnn.benchmark = True #choose best kernel for computation

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")

    train_epoch_loss = fit(model, torch.from_numpy(train_data), torch.from_numpy(y[:-2000]), loss)
    val_epoch_loss = validate(model, torch.from_numpy(val_data), torch.from_numpy(y[-2000:]), loss)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")


# Evaluate

# In[248]:


out = model.forward( torch.from_numpy(val_data))
out = torch.round(out) #round to nearest 0 or 1
out


# In[256]:


len(y[-2000:])


# In[259]:


loss(out, torch.from_numpy(y[-2000:]))


# In[261]:


# Number of wrong elements
val_labels = y[-2000:]
wrong = 0
for i in range(len(val_labels)):
    if(out[i] != val_labels[i]):
        wrong += 1
        
print(wrong)


# In[118]:


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


# In[274]:


ty = torch.from_numpy(y[-2000:])
r2_loss(out, torch.from_numpy(y[-2000:]))


# SVM
# ---
# 10 Fold CV with SVM
# 

# In[17]:


k = 10
kf = KFold(n_splits=k, random_state=None)

avg_acc = []
model = svm.SVC()
acc_score = []


# In[84]:


for train_index , test_index in kf.split(X_new):
    X_train , X_test = X_new[train_index,:],X_new[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)

    acc = np.sqrt(mean_squared_error(pred_values , y_test))
    acc_score.append(acc)

avg_acc_score = sum(acc_score)/k
avg_acc.append(avg_acc_score)


# In[85]:


avg_acc


# Alles false setzen

# In[86]:


1-y.sum()/y.size


# Neue Features
# ---
# Zähle wieviele Male, dass ein Test gemacht wurde und nutze diese Matrix

# In[20]:


train_feat_regroup = train_feat.groupby('pid').agg(lambda x: x.count()).reset_index()


# In[21]:


train_feat_regroup.pop("Age")
train_feat_regroup.pop("Time")


# In[22]:


train_feat_regroup.pop("pid")


# In[23]:


train_feat_regroup


# In[24]:


train_feat_regroupNP = train_feat_regroup.to_numpy()


# SVM 10 CV

# In[53]:


from sklearn.metrics import r2_score
k = 10
kf = KFold(n_splits=k, random_state=None)

avg_acc = []
#model = svm.SVC()
acc_score = []

y = labels[:, 0] #try to predict one label


# In[63]:


for train_index , test_index in kf.split(train_feat_regroup):
    model = svm.SVC()
    X_train , X_test = train_feat_regroupNP[train_index,:], train_feat_regroupNP[test_index,:]
    #X_train , X_test = train_feat_new2[train_index,:], train_feat_new2[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)

    #acc = np.sqrt(mean_squared_error(pred_values , y_test))
    acc = r2_score(y_test, pred_values)
    acc_score.append(acc)

avg_acc_score = sum(acc_score)/k
avg_acc.append(avg_acc_score)


# In[64]:


acc_score


# In[65]:


avg_acc


# In[66]:


from sklearn.metrics import RocCurveDisplay
y_pred = model.decision_function(X_test)
RocCurveDisplay.from_predictions(y_test, y_pred)
plt.show()


# In[39]:


from sklearn.metrics import roc_auc_score
y_pred = model.decision_function(X_test)
roc_auc_score(y_test, y_pred)


# In[40]:


import pickle

# save
with open('model.pkl','wb') as f:
    pickle.dump(model,f)


# In[40]:


import pickle
# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

clf2.predict(train_feat_regroupNP[0:1])


# Test other labels

# In[60]:


from sklearn.metrics import r2_score
k = 10
kf = KFold(n_splits=k, random_state=None)

avg_acc = []
#model = svm.SVC()
acc_score = []

y = labels[:, 8] #try to predict one label


# In[61]:


for train_index , test_index in kf.split(train_feat_regroup):
    model = svm.SVC()
    X_train , X_test = train_feat_regroupNP[train_index,:], train_feat_regroupNP[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)

    #acc = np.sqrt(mean_squared_error(pred_values , y_test))
    acc = r2_score(y_test, pred_values)
    acc_score.append(acc)

avg_acc_score = sum(acc_score)/k
avg_acc.append(avg_acc_score)
acc_score


# In[62]:


from sklearn.metrics import RocCurveDisplay
y_pred = model.decision_function(X_test)
RocCurveDisplay.from_predictions(y_test, y_pred)
plt.show()


# In[63]:


from sklearn.metrics import roc_auc_score
X_test = train_feat_regroup[:1000]
y_test = labels[:1000, 4]

y_pred = model.predict(X_test)
roc_auc_score(y_test, y_pred)


# Make for multiple classes
#     
#         

# In[28]:


from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_feat_regroupNP)
train_feat_regroupNP2 = scaler.transform(train_feat_regroupNP)
y = labels[:,:10]
model = MultiOutputClassifier(svm.SVC()).fit(train_feat_regroupNP2, y)


# In[29]:


from sklearn.metrics import roc_auc_score
X_test = train_feat_regroupNP2
y_test = labels[:,:10]

y_pred = model.predict(X_test)
roc_auc_score(y_test, y_pred, average=None)


# In[53]:


from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_feat_new2)
train_feat_new3 = scaler.transform(train_feat_new2)
y = labels[:,:10]
model = MultiOutputClassifier(svm.SVC()).fit(train_feat_new3, y)


# In[69]:


from sklearn.metrics import roc_auc_score
X_test = train_feat_new3
y_test = labels[:,:10]

y_pred = model.predict(X_test)
roc_auc_score(y_test, y_pred, average=None)


# Histogram-based Gradient Boosting Classification Tree.

# In[74]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay

y = labels[2000:, 0]
X = train_feat_new3[2000:]
clf = HistGradientBoostingClassifier(random_state=0)
clf.fit(X, y)

y_pred = clf.predict_proba(train_feat_new3[:2000])[:,1]
RocCurveDisplay.from_predictions(labels[:2000, 0], y_pred)
plt.show()


# In[81]:


from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_feat_new2)
train_feat_new3 = scaler.transform(train_feat_new2)
y = labels[:,:10]
clf = HistGradientBoostingClassifier(random_state=0)
model = MultiOutputClassifier(clf).fit(train_feat_new3, y)


# In[82]:


from sklearn.metrics import roc_auc_score
X_test = train_feat_new3
y_test = labels[:,:10]

y_pred = model.predict_proba(X_test)
y_pred = np.asarray(y_pred)
y_pred = y_pred[:,:,1]
y_pred = np.transpose(y_pred)
roc_auc_score(y_test, y_pred, average=None)


# In[83]:


scaler = StandardScaler()
scaler.fit(test_feat_new2)
test_feat_new3 = scaler.transform(test_feat_new2)


# In[84]:


y_pred = model.predict_proba(test_feat_new3)
y_pred = np.asarray(y_pred)
y_pred = y_pred[:,:,1]
y_pred = np.transpose(y_pred)

i = 0
for name in ['LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2']:
    Abgabe[name] = y_pred[:,i]
    i += 1


# In[86]:


Abgabe
#ist das korrekt oder müsste es eine 0/1 klassifizierung sein?


# Try Again with Median Imputer

# In[149]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_feat_new2MEDIAN)
train_feat_new3MEDIAN = scaler.transform(train_feat_new2MEDIAN)train_feat_new3MEDIAN


# In[150]:


from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
y = labels[:,:10]
model = MultiOutputClassifier(clf).fit(train_feat_new3MEDIAN, y)

X_test = train_feat_new3MEDIAN
y_test = labels[:,:10]

y_pred = model.predict_proba(X_test)
y_pred = np.asarray(y_pred)
y_pred = y_pred[:,:,1]
y_pred = np.transpose(y_pred)
roc_auc_score(y_test, y_pred, average=None)


# Subtask 2
# ---
# 

# In[90]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0, class_weight="balanced")
model2 = svm.SVC(random_state=0, class_weight="balanced")
cross_val_score(clf, train_feat_regroupNP, labels[:,10],  cv=10)


# In[91]:


clf.fit(train_feat_regroupNP[1000:], labels[1000:,10])


# In[92]:


model2.fit(train_feat_regroupNP[1000:], labels[1000:,10])


# In[99]:


from sklearn.metrics import RocCurveDisplay
y_pred = model2.predict(train_feat_regroupNP[:1000])
RocCurveDisplay.from_predictions(labels[:1000,10], y_pred)
plt.show()


# try with more features and HistGradientBoostingClassifier

# In[88]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.utils.class_weight import compute_class_weight


#y = labels[2000:, 10]
y = labels[:, 10]
wheights = compute_class_weight(class_weight="balanced", classes= [0,1], y=y)
print(wheights)
wheights_data = []
for i in range(18995):
    if y[i] == 0.0:
        wheights_data.append(wheights[0])
    else:
        wheights_data.append(wheights[1])
#print(wheights_data)

#X = train_feat_new3[2000:]
X = train_feat_new3
clf = HistGradientBoostingClassifier(random_state=0)
clf.fit(X, y, sample_weight =wheights_data)

#y_pred = clf.predict_proba(train_feat_new3[:2000])[:,1]
#RocCurveDisplay.from_predictions(labels[:2000, 10], y_pred)
y_pred = clf.predict_proba(train_feat_new3)[:,1]
RocCurveDisplay.from_predictions(labels[:, 10], y_pred) #ust for training to look how good it really is use test/train split
plt.show()


# In[89]:


y_pred = clf.predict_proba(test_feat_new3)[:,1]

i = 0
for name in ['LABEL_Sepsis']:
    Abgabe[name] = y_pred
    i += 1


# In[90]:


Abgabe


# Againg with Median Imputing

# In[151]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.utils.class_weight import compute_class_weight


y = labels[2000:, 10]
wheights = compute_class_weight(class_weight="balanced", classes= [0,1], y=y)
print(wheights)
wheights_data = []
for i in range(16995):
    if y[i] == 0.0:
        wheights_data.append(wheights[0])
    else:
        wheights_data.append(wheights[1])
#print(wheights_data)

X = train_feat_new3MEDIAN[2000:]
clf = HistGradientBoostingClassifier(random_state=0)
clf.fit(X, y, sample_weight =wheights_data)

y_pred = clf.predict_proba(train_feat_new3MEDIAN[:2000])[:,1]
RocCurveDisplay.from_predictions(labels[:2000, 10], y_pred)
plt.show()


# Subtask 3
# ---

# Idee: linear model mit convolution (CNN) der 4 Werte und summe der Tests. Imputing 4 Werte mit durchschnitt zwischen zwei.

# In[92]:


addi_feat = train_feat[['Heartrate','ABPs', 'SpO2' , 'RRate']].to_numpy
addi_feat


# In[89]:


train_feat[['Heartrate','ABPs', 'SpO2' , 'RRate']]


# In[158]:


tf2 = train_feat.interpolate(method='linear', limit_direction="both")
tf2


# In[357]:


tfH = (tf2.groupby(['pid'])['Heartrate']).apply(list).reset_index()
tfH.pop("pid")
tfA = tf2.groupby(['pid'])['ABPs'].apply(list).reset_index()
tfA.pop("pid")
tfS = tf2.groupby(['pid'])['SpO2'].apply(list).reset_index()
tfS.pop("pid")
tfR = tf2.groupby(['pid'])['RRate'].apply(list).reset_index()
tfR.pop("pid")

tfH = tfH.to_numpy()
tfA = tfA.to_numpy()
tfS = tfS.to_numpy()
tfR = tfR.to_numpy()

tf = np.concatenate((tfH, tfA, tfS, tfR), axis=1)
tf.shape


# In[359]:


tf = tf.tolist()
tf = np.asarray(tf)
tf.shape


# In[122]:


#df['ABPs'].interpolate(method='linear', inplace=True, limit_direction="both")


# In[360]:


fulltfNP = np.empty((18995,0))
for name, values in tf2.iteritems():
    print(name)
    if(name == 'pid' or name == 'Time' or name == 'Age'):
        continue
    
    nparr = tf2.groupby(['pid'])[name].apply(list).reset_index()
    nparr.pop("pid")
    nparr = nparr.to_numpy()
    #print(fulltfNP.shape)
    #print(nparr.shape)
    fulltfNP = np.concatenate((fulltfNP, nparr), axis=1)

fulltf = fulltfNP.tolist()
fulltf = np.asarray(fulltf)
fulltf.shape


# Model

# In[366]:


class CNNfull(nn.Module):
    def __init__(self, out_Dim):
        super(CNNfull, self).__init__()
        
        self.conv1 = nn.Conv1d(34, 64, kernel_size=3, stride=2, padding=1)  # 12 x 1
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1)  # 6 x 1 -> 4x1
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)  # 4 x 1 -> 2x1
        
        self.lin0 = nn.Linear(in_features=256, out_features=1024)
        self.lin1 = nn.Linear(in_features=1024, out_features=out_Dim)
        
        self.act = nn.ReLU(inplace=True)
        
 
    def forward(self, x):
        #print(f"inp conv: {x.size()}")
        x = self.conv1(x)
        #print(f"conv1: {x.size()}")
        x = self.act(x)
        x = self.conv2(x)
        #print(f"conv2: {x.size()}")
        x = self.act(x)
        x = self.act(self.conv3(x)).view(-1, 256)
        #print(f"conv3: {x.size()}")
        x = self.act(self.lin0(x))
        values = self.act(self.lin1(x))
        return values


# In[309]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=2, padding=1)  # 12 x 1
        self.conv2 = nn.Conv1d(32, 32, kernel_size=2, stride=2, padding=1)  # 6 x 1 -> 4x1
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)  # 4 x 1 -> 2x1
        
        self.act = nn.ReLU(inplace=True)
        
 
    def forward(self, x):
        #print(f"inp conv: {x.size()}")
        x = self.conv1(x)
        #print(f"conv1: {x.size()}")
        x = self.act(x)
        x = self.conv2(x)
        #print(f"conv2: {x.size()}")
        x = self.act(x)
        values = self.act(self.conv3(x))
        #print(f"conv3: {values.size()}")
        return values


# In[328]:


class Net(nn.Module):
    def __init__(self, in_Dim, out_Dim):
        super(Net, self).__init__()
        
        self.cnn = CNN()
        
        self.lin0 = nn.Linear(in_features=in_Dim, out_features=1024)
        self.lin1 = nn.Linear(in_features=1024, out_features=out_Dim)
        
        self.act = nn.ReLU(inplace=True)

        
    def forward(self, x4, xRest):
        
        c = self.cnn(x4).view(-1, 64)
        #print(f"c: {c.size()}")
        x = torch.cat((xRest, c), axis=1)
        x = self.lin0(x)
        x = self.act(x)
        return self.lin1(x)


# In[329]:


epochs = 12
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[367]:


#model = Net(94, 1).to(device) #labels.shape[1] ).to(device)
model = CNNfull(1).to(device)
print(model)


# In[368]:


opt = optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()


# In[276]:


y = labels[:, -1] #try to predict one label (heartrate)


# In[300]:


train_data4 = tf[:-2000]
val_data4 = tf[-2000:]

train_data = train_feat_regroup.loc[ : , train_feat_regroup.columns != 'Heartrate']
train_data = train_data.loc[ : , train_data.columns != 'ABPs']
train_data = train_data.loc[ : , train_data.columns != 'SpO2']
train_data = train_data.loc[ : , train_data.columns != 'RRate'].to_numpy()
train_dataRest = train_data[:-2000]
val_dataRest = train_data[-2000:]
print(len(train_data4))
print(len(train_dataRest))


train_data4 = train_data4.astype(dtype=np.float32)
val_data4 = val_data4.astype(dtype=np.float32)
train_dataRest = train_dataRest.astype(dtype=np.float32)
val_dataRest = val_dataRest.astype(dtype=np.float32)

y = y.astype(dtype=np.float32)
print(train_data4.dtype)
print(val_data4.dtype)


# In[267]:


def fit(model, X4, XRest, labels, loss):
    X4.to(device)
    XRest.to(device)
    labels.to(device)
    model.train
    running_loss = 0.0
    indexEnd = 64
    indexStart = 0
    train_loss = []
   
    while (True):
        X4t = X4[indexStart:indexEnd]
        XRestt = XRest[indexStart:indexEnd]
        #print(f"X4: {X4t.size()}")
        #print(f"XRestt: {XRestt.size()}")
        
        opt.zero_grad()

        inp = model(X4t, XRestt)        

        output = loss(inp, labels)

        #running_loss += output.item()
        output.backward()
        train_loss.append(output.item())
        #for param in model.parameters():    
            #param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=None, neginf=None)
            #print(param.grad)

        opt.step()
        
        
        if(not(indexEnd < XRest.size(0)-1)):
            break;
        indexEnd += 64
        indexStart += 64
        indexEnd = min(indexEnd, XRest.size(0)-1)
        
    
    return np.array(train_loss).mean()


# In[374]:


def validate(model, X4, XRest, labels, loss):
    X4.to(device)
    XRest.to(device)
    labels.to(device)
    model.eval()
    running_loss = 0.0
    val_loss = []
    
    with torch.no_grad():
        #for i in range(len(labels)):
       
        opt.zero_grad()

        inp = model(X4, XRest)        

        output = loss(inp, labels)

        #running_loss += output.item()
        val_loss.append(output.item())
        
    return np.array(val_loss).mean()


# In[327]:


train_loss = []
val_loss = []
torch.backends.cudnn.benchmark = True #choose best kernel for computation

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")

    train_epoch_loss = fit(model, torch.from_numpy(train_data4), torch.from_numpy(train_dataRest), torch.from_numpy(y[:-2000]), loss)
    val_epoch_loss = validate(model, torch.from_numpy(val_data4), torch.from_numpy(val_dataRest), torch.from_numpy(y[-2000:]), loss)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")


# In[333]:


testIndex = 10
testData4, testDataRest = torch.from_numpy(train_data4[testIndex]), torch.from_numpy(train_dataRest[testIndex])
testData4, testDataRest = testData4[None,:,:], testDataRest[None,:]
#print(testData4.size())
inp = model(testData4, testDataRest) 
print(f"inp: {inp}")
print(f"groundtruth: {y[testIndex]}")


# In[ ]:





# In[369]:


train_dataFull = fulltf[:-2000]
val_dataFull = fulltf[-2000:]
train_dataFull = train_dataFull.astype(dtype=np.float32)
val_dataFull = val_dataFull.astype(dtype=np.float32)
y = y.astype(dtype=np.float32)


# In[370]:


def fitFull(model, X, labels, loss):
    X.to(device)
    labels.to(device)
    model.train
    running_loss = 0.0
    indexEnd = 64
    indexStart = 0
    train_loss = []
   
    while (True):
        Xt = X[indexStart:indexEnd]
        #print(f"X: {X.size()}")
        
        opt.zero_grad()

        inp = model(Xt)        

        output = loss(inp, labels)

        #running_loss += output.item()
        output.backward()
        train_loss.append(output.item())
        #for param in model.parameters():    
            #param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=None, neginf=None)
            #print(param.grad)

        opt.step()
        
        
        if(not(indexEnd < X.size(0)-1)):
            break;
        indexEnd += 64
        indexStart += 64
        indexEnd = min(indexEnd, X.size(0)-1)
        
    
    return np.array(train_loss).mean()


# In[373]:


def validateFull(model, X, labels, loss):
    X.to(device)
    labels.to(device)
    model.eval()
    running_loss = 0.0
    val_loss = []
    
    with torch.no_grad():
        #for i in range(len(labels)):
       
        opt.zero_grad()

        inp = model(X)        

        output = loss(inp, labels)

        #running_loss += output.item()
        val_loss.append(output.item())
        
    return np.array(val_loss).mean()


# In[375]:


train_loss = []
val_loss = []
torch.backends.cudnn.benchmark = True #choose best kernel for computation

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")

    train_epoch_loss = fitFull(model, torch.from_numpy(train_dataFull), torch.from_numpy(y[:-2000]), loss)
    val_epoch_loss = validateFull(model, torch.from_numpy(val_dataFull), torch.from_numpy(y[-2000:]), loss)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")


# In[377]:


testIndex = 100
testData = torch.from_numpy(train_dataFull[testIndex])
testData = testData[None,:]
#print(testData4.size())
inp = model(testData) 
print(f"inp: {inp}")
print(f"groundtruth: {y[testIndex]}")


# Try it again with a HistGradientBoosting 

# In[91]:


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
featIndex = 14

model = HistGradientBoostingRegressor(max_depth=4, random_state=0)
#y = labels[2000:, featIndex]
#X = train_feat_new2[2000:]#unscaled features #for training
y = labels[:, featIndex]
X = train_feat_new2#unscaled features
model.fit(X, y)

y_pred = model.predict(train_feat_new2[:2000])
loss = r2_score(labels[:2000, featIndex], y_pred)
loss


# make multioutput

# In[92]:


from sklearn.multioutput import MultiOutputRegressor
y = labels[:,11:]
model = HistGradientBoostingRegressor(max_depth=4, random_state=0)
model = MultiOutputRegressor(model).fit(train_feat_new2, y)


# In[93]:


y_pred = model.predict(train_feat_new2[:2000])
loss = r2_score(labels[:2000, 11:], y_pred, multioutput='raw_values')
loss


# In[94]:


y_pred = model.predict(test_feat_new2)

i = 0
for name in ['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']:
    Abgabe[name] = y_pred[:,i]
    i += 1


# In[97]:


Abgabe


# Same with Median Imputer

# In[152]:


from sklearn.multioutput import MultiOutputRegressor
y = labels[:,11:]
model = HistGradientBoostingRegressor(max_depth=4, random_state=0)
model = MultiOutputRegressor(model).fit(train_feat_new2MEDIAN, y)


# In[153]:


y_pred = model.predict(train_feat_new2MEDIAN[:2000])
loss = r2_score(labels[:2000, 11:], y_pred, multioutput='raw_values')
loss


# Schlussforlgerung: Mit KNN Imputer und HistGradientBoosting und nicht SVM

# In[96]:


import pandas as pd

# suppose df is a pandas dataframe containing the result
Abgabe.to_csv('prediction1.zip', index=False, float_format='%.3f', compression='zip')


# In[ ]:




