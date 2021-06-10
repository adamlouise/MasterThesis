#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:23:37 2021

@author: louiseadam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


def create_Net1(params):
    num_in = 552
    num_out = 6
    num_h1 = params["num_h1"]
    num_h2 = params["num_h2"]
    num_h3 = params["num_h3"] 
    num_h4 = params["num_h4"]
    num_h5 = params["num_h5"]
    drop_prob = params["dropout"]
    
    net1 = Net1(num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_out, drop_prob)
    
    return net1
    

class Net1(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_out, drop_prob):
        super(Net1, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))
        self.l1_bn = nn.BatchNorm1d(num_h1)
        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        self.l2_bn = nn.BatchNorm1d(num_h2)
        #second hidden layer
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h3, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_h3), 0))
        
        self.W_4 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h4, num_h3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_h4), 0))
        
        self.W_5 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h5, num_h4)))
        self.b_5 = Parameter(init.constant_(torch.Tensor(num_h5), 0))
        
        self.W_6 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h5)))
        self.b_6 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        #self.W_3_bn = nn.BatchNorm2d(num_out)
        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)

        x = F.linear(x, self.W_2, self.b_2)
        #x = self.l1_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_4, self.b_4)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_5, self.b_5)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_6, self.b_6)

        return x