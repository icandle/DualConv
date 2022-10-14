# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


class DualConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            nonlinear=None,
            padding=None,
            deployed = False):
        super(DualConv, self).__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding      
        self.groups = groups
        
        conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          groups=groups,
                          bias=True)
                    
        self.k1 = conv1.weight
        self.b1 = conv1.bias


        conv2 = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          groups=groups,
                          bias=True)    
        
        self.k2 = conv2.weight
        self.b2 = conv2.bias
        
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nn.PReLU(out_channels)

    def get_corr(self, fake_Y, Y):
        fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
        fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
        corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
                    torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
        return corr
 
    def dis_loss(self, x, y):
        x = x.view(1, -1)
        y = y.view(1, -1) 
        c = self.get_corr(x, y)

        return (c+1)/2
    
    def forward(self, x):     
        if self.training:
            cl = self.dis_loss(self.k1,self.k2)
            out = F.conv2d(input=x, weight=self.k1+self.k2, bias=self.b1+self.b2, stride=1, padding=self.padding, groups=self.groups)
            return self.nonlinear(out) , cl
        else:
            out = F.conv2d(input=x, weight=self.k1+self.k2, bias=self.b1+self.b2, stride=1, padding=self.padding, groups=self.groups)
            return self.nonlinear(out) 

