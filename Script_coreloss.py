# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from utils_ae import lr_scheduler_choose
from coreloss import AdaWeightedLoss
from torch.nn import MSELoss
from MLP_force import AutoEncoder

## setting coreloss
'''
'--temperature', type=float, default=0.5,help='loss concentration'
'--hard_ratio', type=float, default=0,help='loss ratio without gradient'
'--direct', type=bool, default=False, help='if gradually use loss'
'--strategy', type=str, default='linear',choices=['exp','linear'], help='gradually strategy'
'''
loss_function='CoreLoss'# MSE MAE CoreLoss
strategy = 'linear'
temperature = 0.5
hard_ratio = 0
direct = False


## load data
signal = scipy.io.loadmat('designal.mat')['signal']
signal = signal[:, np.newaxis, :]
signal = np.repeat(signal,100,axis=0)
noise = np.random.normal(loc=0, scale=10, size=signal.shape)
signal = signal+noise

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
        self.type = type
    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        elif self.type=='None':
            seq=seq
        elif self.type=='mean':
            seq = seq - seq.mean()
        else:
            raise NameError('This normalization is not included!')
        return seq

Normalization = Normalize('1-1')
signal = Normalization(signal)

class SigDataset(Dataset):
    def __init__(self, signal):
        super(SigDataset,self).__init__()
        self.signal=torch.tensor(signal,dtype=torch.float32)
    def __len__(self):
        return len(self.signal)
    def __getitem__(self, idx):
        return self.signal[idx]

def LoadSig(signal,batch_size=16):
    dataset=SigDataset(signal)
    print('{} samples found'.format(len(dataset)))
    train_iterator=DataLoader(dataset,batch_size,shuffle=False)
    return train_iterator


if loss_function == 'MSE':
    reconstruction_loss = MSELoss()  # 训练ae
elif loss_function == 'MAE':
    reconstruction_loss = nn.L1Loss()  # 训练ae
elif loss_function == 'CoreLoss':
    reconstruction_loss = AdaWeightedLoss(strategy=strategy,temp=temperature,hard_ratio=hard_ratio,direct=direct)

epochs =10
device='cuda'
model=AutoEncoder(inp_dim=1, z_dim=20, seqlen=4096)
model.cuda()
# opt = torch.optim.Adam(params=model.parameters() ,lr=0.001, betas=(0.9,0.999), eps=1e-07)
opt = torch.optim.SGD(params=model.parameters() ,lr=0.001, momentum=0.9)
lr_scheduler = lr_scheduler_choose(lr_scheduler_way='step',optimizer=opt,steps='20,50',gamma=0.1)


train_iter =LoadSig(signal) #用污染的数据训练
cur_step = 0
for epoch in range(epochs):
    running_loss = 0.0
    all_number = 0
    model.train()
    # train
    for i, sig in enumerate(train_iter):
        cur_step += 1
        if epoch%300==0 and i==0:
            print('current lr: {}'.format(lr_scheduler.get_lr()))
        opt.zero_grad()
        sig = sig.cuda()
        Re_sig, z_hidden = model(sig)
        if loss_function == 'CoreLoss':
            re_loss = reconstruction_loss(Re_sig, sig, cur_step)
        else:
            re_loss = reconstruction_loss(Re_sig, sig)

        re_loss.backward()
        opt.step()
        running_loss += re_loss.item()*sig.size()[0]
        all_number += sig.size()[0]
    lr_scheduler.step()
    if epoch%300==0:
        print('[epoch:%d/%d] loss: %.3f'
          % (epoch+1, epochs, running_loss/all_number))

