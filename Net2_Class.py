#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:31:25 2021

@author: louiseadam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

def create_Net2(params):
    
    num_fasc = 2
    num_atoms = 782
    num_w_out = params["num_w_out"] 
    num_w_l1 = params["num_w_l1"]
    num_w_l2 = params["num_w_l2"]
    num_w_l3 = params["num_w_l3"]
    num_w_in = num_atoms
    num_f_out = 6 #nombre de paramètres à estimer
    num_f_l1 = params["num_f_l1"]
    num_f_l2 = params["num_f_l2"]
    num_f_in = num_w_out*num_fasc #ici 10*2
    drop = params["dropout"]
    
    net_tot = Net_tot(num_w_in, num_w_l1, num_w_l2, num_w_l3, num_w_out, num_f_in, num_f_l1, num_f_l2, num_f_out, drop)
    
    return net_tot


# Network 1

class Net_w(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_out, drop_prob):
        super(Net_w, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h3, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_h3), 0))
        
        self.W_4 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)

        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_4, self.b_4)

        return x


# Network 2
class Net_f(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_out, drop_prob):
        super(Net_f, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.linear(x, self.W_3, self.b_3)

        return x

# Network 3
class Net_tot(nn.Module):

    def __init__(self, numw_in, numw_l1, numw_l2, numw_l3, numw_out, numf_in, numf_l1, numf_l2, numf_out, drop):
        super(Net_tot, self).__init__()  
        self.netw = Net_w(numw_in, numw_l1, numw_l2, numw_l3, numw_out, drop_prob=drop)
        self.netf = Net_f(numf_in, numf_l1, numf_l2, numf_out, drop_prob=drop)

    def forward(self, w1, w2):
        x1 = self.netw(w1)
        x2 = self.netw(w2)
        
        x = torch.cat((x1, x2), axis=1)

        x = self.netf(x)

        return x
