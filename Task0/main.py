from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


device = 'cpu'

train = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task0\task0_sl19d1\train.csv')
test = pd.read_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task0\task0_sl19d1\test.csv')

train = torch.Tensor(train.to_numpy(), device=device) 
test = torch.as_tensor(test.to_numpy(), device=device) 
print(train.size())
print(train[0:2])
print(test.size())
print(test[0:2])


#train = train.view(-1, 1, 1, 10) 
trainID, trainY, trainX = torch.split(train, (1, 1, 10), dim=1)#items, ID, Y, X1-X10
testID, testX = torch.split(test, (1,10), dim = 1)#items, ID, X1-X10
print(trainX.size())
print(trainX[0:3])

#meanX = torch.nn.functional.normalize(trainX, dim=0, p=1.0)
meanX = trainX.mean(dim=1)
print(meanX[0])
print(trainY[0])
print((meanX-trainY).mean())


testMean = testX.mean(dim=1).view(-1,1)
print(testMean.size())
print(testID.size())
subm = torch.cat((testID, testMean), dim=1)
print(subm.size())
print(subm[0])

d = {'Id': testID.view(-1), 'y': testMean.view(-1)}

df = pd.DataFrame(data=d, index = None)
print(df)
df.to_csv(r'C:\Users\erics\Documents\Programme\IntroML\Task0\submission.csv', index=False)
