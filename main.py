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
from net.ssd.net_tool import get_default_boxes
from data import TrainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    cfg._print()
    # net=SSD(21)
    # net._print()
    # im=torch.randn([1,3,300,300])
    # for loc,cls in net(im):
        # print(loc.shape,cls.shape)
    # for k,v in net.named_parameters():
    #     print(k,v,v.shape)

    # boxes=get_default_boxes()
    # print(boxes.shape)
    data_set=TrainDataset()

    data_loader=DataLoader(
        data_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_worker
    )

    for i,(imgs,targets,labels) in tqdm(enumerate(data_loader)):
        print(imgs.shape,targets.shape,labels.shape)

if __name__ == '__main__':
    main()