# coding=UTF-8
# author=theppsh

from config import cfg
import numpy as np 
import torch

# hold the rand_seed...
np.random.seed(cfg.rand_seed)
torch.manual_seed(cfg.rand_seed)
torch.cuda.manual_seed(cfg.rand_seed)

from net import SSD
def main():
    cfg._print()
    net=SSD(21)
    net._print()
    # for k,v in net.named_parameters():
    #     print(k,v,v.shape)

if __name__ == '__main__':
    main()