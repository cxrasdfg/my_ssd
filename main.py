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
from data import TrainDataset,eval_net,get_check_point
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

def train():
    cfg._print()
   
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
    
    # NOTE: plus one, en?
    net=SSD(len(data_set.classes)+1)
    net._print()
    epoch,iteration,w_path=get_check_point()
    if w_path:
        model=torch.load(w_path)
        net.load_state_dict(model)
        print("Using the model from the last check point:%s"%(w_path),end=" ")
        epoch+=1

    net.train()
    is_cuda=cfg.use_cuda
    did=cfg.device_id
    if is_cuda:
        net.cuda(did)

    while epoch<cfg.epochs:
        
        print('********\t EPOCH %d \t********' % (epoch))
        for i,(imgs,targets,labels) in tqdm(enumerate(data_loader)):
            if is_cuda:
                imgs=imgs.cuda(did)
                targets=targets.cuda(did)
                labels.cuda(did)

            _loss=net.train_once(imgs,targets,labels)
            tqdm.write('Epoch:%d, iter:%d, loss:%.5f'%(epoch,iteration,_loss))

            iteration+=1

        torch.save(net.state_dict(),'%sweights_%d_%d'%(cfg.weights_dir,epoch,iteration) )
        epoch+=1

        _map= eval_net(net=net,num=100,shuffle=True)['map']
        print("map:",_map)
        epoch+=1
    print(eval_net(net=net))


if __name__ == '__main__':
    opt=sys.argv[1]
    if opt=='train':
        train()
    elif opt=='test':
        test_net()
    elif opt=='eval':
        print(eval_net())
    else:
        raise ValueError('opt shuold be in [`train`,`test`,`eval`]')
    # train()