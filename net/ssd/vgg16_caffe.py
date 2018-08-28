# coding=UTF-8
import torch
from torchvision.models import vgg16
from config import cfg

def caffe_vgg16():
    base_model=vgg16(False)
    base_model.load_state_dict(torch.load(cfg.caffe_model) )
    # 21 is Conv4-3, 30 is the max pooling
    features=list(base_model.features)[:30] # list...
    conv4_3=features[:21]
    conv5_3=features[21:]
    
    conv4_3=torch.nn.Sequential(*conv4_3)
    conv5_3=torch.nn.Sequential(*conv5_3)

    return conv4_3,conv5_3
    