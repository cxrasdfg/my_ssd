# coding=UTF-8
# author=theppsh

from config import cfg
import numpy as np 
import torch
np.random.seed(cfg.rand_seed)
torch.manual_seed(cfg.rand_seed)
torch.cuda.manual_seed(cfg.rand_seed)

from net import SSD
def main():
    cfg._print()
    SSD()

if __name__ == '__main__':
    main()