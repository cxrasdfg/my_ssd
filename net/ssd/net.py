# coding=UTF-8
import torch
import numpy as np
from config import cfg

from .vgg16_caffe import caffe_vgg16 as vgg16

class ConvBN2d(torch.nn.Module):
    def __init__(self,in_,out_,size_,pad_,stride_,bn_,relu_=True):
        super(ConvBN2d,self).__init__()

        self.conv=torch.nn.Conv2d(in_,out_,size_,stride_,padding=pad_)
        self.bn=torch.nn.BatchNorm2d(out_) if bn_ else None
        self.act=torch.nn.ReLU(inplace=True) if relu_ else None
    
    def forward(self,x):
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        if self.act:
            x=self.act(x)
        return x

class SSD(torch.nn.Module):
    def __init__(self,class_num):
        super(SSD,self).__init__()
        
        # this num should contain the background...
        self.class_num=class_num

        self.conv4,self.conv5,self.conv6,self.conv7=vgg16()

        bn_=cfg.use_batchnorm        
        
        self.conv8=torch.nn.Sequential(*[
            ConvBN2d(1024,256,1,0,1,bn_),
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

        ar=cfg.ar
        feat_map=cfg.feat_map
        det_in=cfg.det_in_channels

        assert len(ar) == len(feat_map)
        assert len(ar) == len(det_in)

        num_loc_channels=[len(_)*2*4 for _ in ar]
        num_cls_channels=[len(_)*2*self.class_num for _ in ar]

        conv_loc_layers=[]
        conv_cls_layers=[]

        for input_ch,num_loc_ch,num_cls_ch in \
            zip(det_in,num_loc_channels,num_cls_channels):
            conv_loc_layers+=[ConvBN2d(input_ch,num_loc_ch,3,1,1,bn_,False)]
            conv_cls_layers+=[torch.nn.Sequential(*[
                ConvBN2d(input_ch,num_cls_ch,3,1,1,bn_,False),
                torch.nn.Softmax(dim=1)
            ])]

        self.conv_loc_layers=torch.nn.Sequential(*conv_loc_layers)
        self.conv_cls_layers=torch.nn.Sequential(*conv_cls_layers)
    
        

    def _print(self):
        print('********\t NET STRUCTURE \t********')
        print(torch.nn.Sequential(*[
            self.conv4,self.conv5,self.conv6,self.conv7,
            self.conv8,self.conv9,self.conv10,self.conv11,
            self.conv_loc_layers,self.conv_cls_layers
        ]))
        print('********\t NET END \t********')
