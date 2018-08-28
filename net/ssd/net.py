# coding=UTF-8
import torch
import numpy as np
from config import cfg

from .vgg16_caffe import caffe_vgg16 as vgg16

class ConvBN2d(torch.nn.Module):
    def __init__(self,in_,out_,size_,pad_,stride_,bn_):
        super(ConvBN2d,self).__init__()

        self.conv=torch.nn.Conv2d(in_,out_,size_,stride_,padding=pad_)
        self.bn=torch.nn.BatchNorm2d(out_) if bn_ else None
        self.act=torch.nn.ReLU(inplace=True)
    
    def forward(self,x):
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        x=self.act(x)
        return x

class SSD(torch.nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        
        self.conv4_3,self.conv5_3,self.conv6,self.conv7=vgg16()

        bn_=cfg.use_batchnorm        
        
        self.conv8=torch.nn.Sequential(*[
            ConvBN2d(self.conv7[-2].out_channels,256,1,0,1,bn_),
            ConvBN2d(256,512,3,1,2,bn_)
        ])
        
        self.conv9=torch.nn.Sequential(*[
            ConvBN2d(512,128,1,0,1,bn_),
            ConvBN2d(128,256,3,1,2,bn_)
        ])

        self.conv10=torch.nn.Sequential(*[
            ConvBN2d(256,128,1,0,1,bn_),
            ConvBN2d(128,256,3,0,1,bn_)
        ])

        self.conv11=torch.nn.Sequential(*[
            ConvBN2d(256,128,1,0,1,bn_),
            ConvBN2d(128,256,3,0,1,bn_)
        ] )

        
        