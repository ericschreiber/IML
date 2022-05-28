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

#device = 'cpu'


# Data to Numpy

# In[2]:


train_feat100 = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task4\task4_hr35z9\train_features.csv')
train_labels100 = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task4\task4_hr35z9\train_labels.csv')
pretrain_feat = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task4\task4_hr35z9\pretrain_features.csv')
pretrain_labels = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task4\task4_hr35z9\pretrain_labels.csv')

test_feat = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task4\task4_hr35z9\test_features.csv')


# In[3]:


test_feat


# In[4]:


Abgabe = pd.DataFrame({'pid': test_feat.iloc[:, 0].values})
Abgabe


# In[5]:


train_feat100.pop("Id")
train_feat100.pop("smiles")
train_labels100.pop("Id")
pretrain_feat.pop("Id")
pretrain_feat.pop("smiles")
pretrain_labels.pop("Id")

test_feat.pop("Id")
test_feat.pop("smiles")


# In[6]:


test_feat


# In[7]:


train_featNP100 = train_feat100.to_numpy(dtype=np.float32)
train_labelsNP100 = train_labels100.to_numpy(dtype=np.float32)
pretrain_featNP = pretrain_feat.to_numpy(dtype=np.float32)
pretrain_labelsNP = pretrain_labels.to_numpy(dtype=np.float32)

test_featNP = test_feat.to_numpy(dtype=np.float32)


# In[8]:


pretrain_feat.shape


# Normalization

# In[9]:


pretrain_featNPnormal = torch.nn.functional.normalize(torch.from_numpy(pretrain_featNP))
train_featNP100normal = torch.nn.functional.normalize(torch.from_numpy(train_featNP100))
test_featNPnormal = torch.nn.functional.normalize(torch.from_numpy(test_featNP))


# Correlation

# In[9]:


Train_feat_corr = train_feat100.corr()


# In[10]:


#Train_feat_corr


# Training
# ---

# In[10]:


class BigNet(nn.Module):
    def __init__(self, in_Dim):
        super(BigNet, self).__init__()
        
        self.lin0 = nn.Linear(in_features=in_Dim, out_features=8000)
        self.lin1 = nn.Linear(in_features=8000, out_features=2048)
        self.lin2 = nn.Linear(in_features=2048, out_features=1024)
        self.lin3 = nn.Linear(in_features=1024, out_features=256)
        self.lin4 = nn.Linear(in_features=256, out_features=10)
        self.lin5 = nn.Linear(in_features=10, out_features=512)
        self.lin6 = nn.Linear(in_features=512, out_features=1)
        
        
        self.BN0 = nn.BatchNorm1d(8000)
        self.BN1 = nn.BatchNorm1d(2048)
        self.BN2 = nn.BatchNorm1d(1024)
        self.BN3 = nn.BatchNorm1d(256)
        self.BN4 = nn.BatchNorm1d(10)
        self.BN5 = nn.BatchNorm1d(512)
    
        self.act = nn.ReLU(inplace=True)

       
        #self.DO = nn.Dropout(p=0.1)
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        
        x = self.act(self.BN0(self.lin0(x)))
        #x = self.DO(x)
        x = self.act(self.BN1(self.lin1(x)))
        
        x = self.act(self.BN2(self.lin2(x)))
        x = self.act(self.BN3(self.lin3(x)))
        x = self.act(self.BN4(self.lin4(x)))
        x = self.act(self.BN5(self.lin5(x)))
        
        return self.lin6(x)
        


# In[11]:


class MediumNet(nn.Module):
    def __init__(self, in_Dim):
        super(MediumNet, self).__init__()
        
        self.lin0 = nn.Linear(in_features=in_Dim, out_features=1024)
        self.lin3 = nn.Linear(in_features=1024, out_features=256)
        self.lin4 = nn.Linear(in_features=256, out_features=1)
        #self.lin6 = nn.Linear(in_features=512, out_features=1)
        
        
        self.BN2 = nn.BatchNorm1d(1024)
        self.BN3 = nn.BatchNorm1d(256)
        #self.BN4 = nn.BatchNorm1d(10)
    
        self.act = nn.ReLU(inplace=True)

       
        #self.DO = nn.Dropout(p=0.1)
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        
        x = self.act(self.BN2(self.lin0(x)))
        x = self.act(self.BN3(self.lin3(x)))
        #x = self.act(self.BN4(self.lin4(x)))
        #x = self.act(self.BN5(self.lin5(x)))
        
        return self.lin4(x)


# In[12]:


class Net(nn.Module):
    def __init__(self, in_Dim):
        super(Net, self).__init__()
        
        #self.lin0 = nn.Linear(in_features=in_Dim, out_features=8000)
        #self.lin1 = nn.Linear(in_features=8000, out_features=2048)
        #self.lin2 = nn.Linear(in_features=2048, out_features=1024)
        #self.lin3 = nn.Linear(in_features=1024, out_features=256)
        #self.lin4 = nn.Linear(in_features=256, out_features=10)
        #self.lin5 = nn.Linear(in_features=10, out_features=512)
        #self.lin6 = nn.Linear(in_features=512, out_features=1)
        
        self.lin0 = nn.Linear(in_features=in_Dim, out_features=750)
        self.lin1 = nn.Linear(in_features=750, out_features=20)
        self.lin2 = nn.Linear(in_features=20, out_features=1)
        
        #self.BN0 = nn.BatchNorm1d(8000)
        #self.BN1 = nn.BatchNorm1d(2048)
        #self.BN2 = nn.BatchNorm1d(1024)
        #self.BN3 = nn.BatchNorm1d(256)
        #self.BN4 = nn.BatchNorm1d(10)
        #self.BN5 = nn.BatchNorm1d(512)
    
        self.BN0 = nn.BatchNorm1d(750)
        self.BN1 = nn.BatchNorm1d(20)        
        self.act = nn.ReLU(inplace=True)
        
        #self.DO = nn.Dropout(p=0.1)
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        
        #x = self.act(self.BN0(self.lin0(x)))
        #x = self.DO(x)
        #x = self.act(self.BN1(self.lin1(x)))
        
        #x = self.act(self.BN2(self.lin2(x)))
        #x = self.act(self.BN3(self.lin3(x)))
        #x = self.act(self.BN4(self.lin4(x)))
        #x = self.act(self.BN5(self.lin5(x)))
        
        
        x = self.act(self.BN0(self.lin0(x)))
        x = self.act(self.BN1(self.lin1(x)))
        return self.lin2(x)


# In[13]:


epochs = 150
batch_size = 16
lr = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[15]:


model = Net(train_feat100.shape[1]).to(device) #labels.shape[1] ).to(device)
print(model)


# In[17]:


from torchsummary import summary
summary(model, input_size=(0,train_feat100.shape[1]))


# In[18]:


from torchmetrics.functional import mean_absolute_percentage_error
opt = optim.Adam(model.parameters(), lr=lr)
loss = mean_absolute_percentage_error  #nn.MSELoss()


# In[19]:


train_data = pretrain_featNPnormal[8000:]
train_labels = pretrain_labelsNP[8000:]
val_data = pretrain_featNPnormal[:8000]
val_labels = pretrain_labelsNP[:8000]


# In[20]:


train_labels.shape


# In[21]:


def fit(model, inpX, inp_labels):
    inpX = inpX#.to(device)
    inp_labels = inp_labels[:,0]#.to(device)
    model.train()
    train_loss = []
    model.to(device)
    for i in range(0, len(inp_labels)-batch_size, batch_size):
        if i % 10000 == 0:
            print(f"i: {i/10000}")   
        opt.zero_grad()
        
        running_loss = 0.0
        X = inpX[i:(i+batch_size)].to(device)
        labels = inp_labels[i:i+batch_size].to(device)
        #print(X.device)

        
        #opt.zero_grad()

        inp = model(X)#.detach().cpu()
        inp = inp.view(inp.shape[0])
        

        
        output = loss(inp, labels)
        #output.retain_grad()
        output.backward()
        opt.step()
        running_loss += output.item()
        

        

        train_loss.append(running_loss)
    return train_loss, model


# In[606]:


def validate(model, X, labels):
    inpX = X#.to(device)
    inp_labels = labels[:,0].to(device)
    model.eval()
    running_loss = 0.0
    val_loss=[]
    
    with torch.no_grad():
        for i in range(0, len(labels)-batch_size, batch_size):
        
        
            running_loss = 0.0
            X = inpX[i:(i+batch_size)].to(device)
            labels = inp_labels[i:i+batch_size].to(device)
            #print(X.device)

            #opt.zero_grad()

            inp = model(X)        
            inp = inp.view(inp.shape[0])
                
            output = loss(inp, labels)



            running_loss += output.item()
            val_loss.append(running_loss)
    
    return val_loss, model


# In[613]:


train_loss = []
val_loss = []

torch.backends.cudnn.benchmark = True #choose best kernel for computation
val_epoch_loss, model = validate(model, val_data, torch.from_numpy(val_labels))
print(f"Val Loss: {np.mean(np.asarray(val_epoch_loss))}")

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")

    train_epoch_loss, model = fit(model, train_data, torch.from_numpy(train_labels))
    val_epoch_loss, model = validate(model, val_data, torch.from_numpy(val_labels))
    print(f"Train Loss: {np.mean(np.asarray(train_epoch_loss))}")
    print(f"Val Loss: {np.mean(np.asarray(val_epoch_loss))}")
    train_loss.append(np.mean(np.asarray(train_epoch_loss)))
    val_loss.append(np.mean(np.asarray(val_epoch_loss)))

    
    
print(f"Train Loss: {train_loss}")
print(f"Val Loss: {val_loss}")


# In[614]:


epochsSt = []
for i in range(epochs):
    epochsSt.append(i+1)


# In[615]:


import matplotlib.pyplot as plt
startRange = 3
stopRange = 500
plt.plot(epochsSt[startRange:stopRange], train_loss[startRange:stopRange])
plt.plot(epochsSt[startRange:stopRange], val_loss[startRange:stopRange])
plt.show()


# In[616]:


from sklearn.metrics import mean_squared_error
model.to('cpu')
y_pred = model(val_data).detach().numpy()
model.to(device)
mean_squared_error(val_labels, y_pred, squared=False)


# In[617]:


print(model.device)


# In[618]:


torch.save(model.state_dict(), 'Task4pretrainedModel')


# In[149]:


model.load_state_dict(torch.load('Task4pretrainedModel'))


# Freeze all except last layer

# In[150]:


for param in model.parameters():
    param.requires_grad = False
for param in model.lin2.parameters():
    param.requires_grad = True


# In[152]:


lrFine= 0.005
epochsFine = 9000
optFine = optim.Adam(model.parameters(), lr=lrFine)
lossFine = nn.MSELoss()


# In[153]:


train_featNP100torch = train_featNP100normal
train_labelsNP100torch = torch.from_numpy(train_labelsNP100)[:,0]


# In[154]:


trainFine = train_featNP100torch#[:85]
trainLabelsFine = train_labelsNP100torch#[:85]
valFine = train_featNP100torch[85:]
valLabelsFine = train_labelsNP100torch[85:]


# In[155]:


train_labelsNP100torch.shape


# In[156]:


#epochsFine = 100
#train_loss = []
#for i in range(epochsFine):
#    print(f"epoch {i}")
    
#    opt.zero_grad()

#    running_loss = 0.0
#    X = train_featNP100torch.to(device)
#    labels = train_labelsNP100torch.to(device)
    #print(X.device)


    #opt.zero_grad()

#    inp = model(X)#.detach().cpu()
#    inp = inp.view(inp.shape[0])



#    output = lossFine(inp, labels)
    #output.retain_grad()
#    output.backward()
#    optFine.step()
#    running_loss += output.item()
    


#    print(f"Train Loss: {running_loss}")
#    train_loss.append(running_loss)
    
#print(f"Train Loss: {np.mean(np.asarray(train_loss))}")


# In[157]:


def fit2(model, inpX, inp_labels):
    inpX = inpX.to(device)
    inp_labels = inp_labels.to(device)
    model.train()
    model.to(device)
    
    
    opt.zero_grad()


    opt.zero_grad()

    inp = model(inpX)#.detach().cpu()
    inp = inp.view(inp.shape[0])
    

    output = lossFine(inp, inp_labels)
    output.backward()
    optFine.step()

    return output.item(), model


# In[158]:


def validate2(model, X, labels):
    inpX = X.to(device)
    inp_labels = labels.to(device)
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        
        #opt.zero_grad()

        inp = model(inpX)        
        inp = inp.view(inp.shape[0])
        
        output = loss(inp, inp_labels)

    return output.item(), model


# In[159]:


train_loss = []
val_loss = []

torch.backends.cudnn.benchmark = True #choose best kernel for computation

for epoch in range(epochsFine):
    print(f"Epoch {epoch+1} of {epochsFine}")

    train_epoch_loss, model = fit2(model, trainFine, trainLabelsFine)
    val_epoch_loss, model = validate2(model, valFine, valLabelsFine)
    print(f"Train Loss: {np.mean(np.asarray(train_epoch_loss))}")
    print(f"Val Loss: {np.mean(np.asarray(val_epoch_loss))}")
    train_loss.append(np.mean(np.asarray(train_epoch_loss)))
    val_loss.append(np.mean(np.asarray(val_epoch_loss)))

print(f"Train Loss: {train_loss}")
print(f"Val Loss: {val_loss}")


# In[160]:


epochsSteps = []
for i in range(epochsFine):
    epochsSteps.append(i+1)


# In[162]:


import matplotlib.pyplot as plt
startRange = 300
stopRange = 15000
plt.plot(epochsSteps[startRange:stopRange], train_loss[startRange:stopRange])
plt.plot(epochsSteps[startRange:stopRange], val_loss[startRange:stopRange])
plt.show()


# In[163]:


model.eval()
model.cpu()
yTest_pred = model(test_featNPnormal).detach().cpu().numpy()


# In[164]:


yTest_pred=yTest_pred[:,0]
yTest_pred.shape


# In[166]:


dictionary = {'y': yTest_pred}
df = pd.DataFrame(dictionary)
AbgabeFertig = pd.concat([Abgabe, df], axis=1)
AbgabeFertig=AbgabeFertig.rename(columns={"pid":"Id"})
print(AbgabeFertig)
AbgabeFertig.to_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task4\submissionV1.csv', index=False, header=True)


# In[ ]:




