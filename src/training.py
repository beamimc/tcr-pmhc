import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score

from model import Net
from loader import Tcr_pMhc_Dataset

from collections import namedtuple
NetParams = namedtuple('NetParams', ['lr',
                                     'wd',
                                     'dropout',
                                     'lstm_dim',
                                     'aa_embedding_dim', 
                                     'mhc_embedding_dim'
                                    ])

###############################
###    Load data            ###
###############################

data_list = []
target_list = []

for fp in glob.glob("data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    
    data_list.append(data)
    target_list.append(targets)

# Note:
# Choose your own training and val set based on data_list and target_list
# Here using the last partition as val set

X_train = np.concatenate(data_list[ :-1])
y_train = np.concatenate(target_list[:-1])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples,nx,ny)

X_val = np.concatenate(data_list[-1: ])
y_val = np.concatenate(target_list[-1: ])
nsamples, nx, ny = X_val.shape
print("val set shape:", nsamples,nx,ny)

p_neg = len(y_train[y_train == 1])/len(y_train)*100
print("Percent positive samples in train:", p_neg)

p_pos = len(y_val[y_val == 1])/len(y_val)*100
print("Percent positive samples in val:", p_pos)


X_train_aa=[]
for i, seq in enumerate(X_train):
  aux1= []
  for j in range(0,420):
    aux1.append(X_train[i][j][:20])
  X_train_aa.append(aux1)


X_val_aa=[]
for i, seq in enumerate(X_val):
  aux1= []
  for j in range(0,420):
    aux1.append(X_train[i][j][:20])
  X_val_aa.append(aux1)


# X_train_div= []
# for i, seq in enumerate(X_train_aa):
#   X_train_div.append([])
#   mhc = seq[:179]
#   p = seq[179:192]
#   tcr = seq[192:-1]
#   X_train_div[i].append(mhc)
#   X_train_div[i].append(p)
#   X_train_div[i].append(pcr)

# X_val_div= []
# for i, seq in enumerate(X_val_aa):
#   X_val_div.append([])
#   mhc = seq[:179]
#   p = seq[179:192]
#   tcr = seq[192:-1]
#   X_val_div[i].append(mhc)
#   X_val_div[i].append(p)
#   X_val_div[i].append(pcr)

# make the data set into one dataset that can go into dataloader
# train_ds = Tcr_pMhc_Dataset(X_train_div, y_train)
# val_ds = Tcr_pMhc_Dataset(X_val_div, y_val)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train_aa)):
    train_ds.append([np.transpose(X_train_aa[i]), y_train[i]])

val_ds = []
for i in range(len(X_val_aa)):
    val_ds.append([np.transpose(X_val_aa[i]), y_val[i]])

bat_size = 64
print("\nNOTE:\nSetting batch-size to", bat_size)
train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
val_ldr = torch.utils.data.DataLoader(val_ds,batch_size=bat_size, shuffle=True)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
#device = torch.device("cpu")



###############################
###    Define network       ###
###############################

print("Initializing network")

# Hyperparameters
input_size = 420
num_classes = 1
learning_rate = 0.01
   
# Initialize network
net_params = NetParams(1e-4, 0, 0.1, 500, 10, 50)
print(net_params)
net = Net(net_params).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)



###############################
###         TRAIN           ###
###############################

print("Training")

num_epochs = 5

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
losses = []
val_losses = []

for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    
    net.train()
    train_preds, train_targs = [], [] 
    for batch_idx, (data, target) in enumerate(train_ldr):
        print(data)
        tcr_batch = data[2].float().detach().requires_grad_(True)
        pep_batch = data[1].float().detach().requires_grad_(True)
        mhc_batch = data[0].float().detach().requires_grad_(True)
        print(len(data))
        print(len(tcr_batch))
        print(len(pep_batch))
        print(len(mhc_batch))
        target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1)
        
        optimizer.zero_grad()
        output = net(tcr_batch, pep_batch, mhc_batch)
        
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        preds = np.round(output.detach().cpu())
        train_targs += list(np.array(target_batch.cpu()))
        train_preds += list(preds.data.numpy().flatten())
        cur_loss += batch_loss.detach()

    losses.append(cur_loss / len(train_ldr.dataset))
        
    
    net.eval()
    ### Evaluate validation
    val_preds, val_targs = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_ldr): ###
            x_batch_val = data.float().detach()
            y_batch_val = target.float().detach().unsqueeze(1)
            
            output = net(x_batch_val)
            
            val_batch_loss = criterion(output, y_batch_val)
            
            preds = np.round(output.detach())
            val_preds += list(preds.data.numpy().flatten()) 
            val_targs += list(np.array(y_batch_val))
            val_loss += val_batch_loss.detach()
            
        val_losses.append(val_loss / len(val_ldr.dataset))
        print("\nEpoch:", epoch+1)
        
        train_acc_cur = accuracy_score(train_targs, train_preds)  
        valid_acc_cur = accuracy_score(val_targs, val_preds) 

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        from sklearn.metrics import matthews_corrcoef
        print("Training loss:", losses[-1].item(), "Validation loss:", val_losses[-1].item(), end = "\n")
        print("MCC Train:", matthews_corrcoef(train_targs, train_preds), "MCC val:", matthews_corrcoef(val_targs, val_preds))
        
print('\nFinished Training ...')

# Write model to disk for use in predict.py
print("Saving model to src/model.pt")
torch.save(net.state_dict(), "src/model.pt")
