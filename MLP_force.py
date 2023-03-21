#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import os
from time import time
from copy import deepcopy
import scipy.io
import torch
from torch import optim
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import logging
import matplotlib.pyplot as plt
from utils_ae import lr_scheduler_choose

from utils_ae import anomaly_scoring
class baseline_Encoder(nn.Module):
    """
    An implementation of Encoder based on MLP
    #inp_dim=channel z_dim=dimension of latent code
    """
    def __init__(self, inp_dim=1, z_dim=8, seqlen=1024):
        super(baseline_Encoder, self).__init__()

        self.linear1 = nn.Linear(seqlen, 64)  #1024->64
        self.linear2 = nn.Linear(64, 64)  #64->64
        self.linear3 = nn.Linear(64, z_dim)  #64->8
        self.relu=nn.ReLU()

    def forward(self, x):
        # inp shape: [bsz, inp_dim, seq_len]
        x=self.linear1(x)
        x=self.relu(x)
        x =self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

class baseline_Decoder(nn.Module):
    def __init__(self, inp_dim, z_dim, seqlen):
        super(baseline_Decoder, self).__init__()

        self.reconstruct = nn.Linear(64,seqlen)  # 1024->64
        self.linear2 = nn.Linear(64, 64)  # 64->64
        self.linear3 = nn.Linear(z_dim,64)  # 64->8
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, inp):

        x = self.linear3(inp)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.reconstruct(x)
        return x

class AutoEncoder(nn.Module):#inp_dim=channel z_dim=10
    def __init__(self, inp_dim, z_dim,seqlen):
        super(AutoEncoder, self).__init__()
        self.encoder = baseline_Encoder(inp_dim, z_dim,seqlen)
        self.decoder = baseline_Decoder(inp_dim, z_dim,seqlen)
    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        z = self.encoder(inp)
        re_inp = self.decoder(z)
        return re_inp, z
