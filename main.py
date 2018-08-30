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

def test_net():
    data_set=TestDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)

    classes=data_set.classes
    net=MyNet(classes)
    _,_,last_time_model=get_check_point()
    # assign directly
    # last_time_model='./weights/weights_21_110242'

    if os.path.exists(last_time_model):
        model=torch.load(last_time_model)
        if cfg.test_use_offline_feat:
            net.load_state_dict(model)
        else:
            net.load_state_dict(model)
        print("Using the model from the last check point:`%s`"%(last_time_model))
    else:
        raise ValueError("no model existed...")
    net.eval()
    is_cuda=cfg.use_cuda
    did=cfg.device_id
    # img_src=cv2.imread("/root/workspace/data/VOC2007_2012/VOCdevkit/VOC2007/JPEGImages/000012.jpg")
    # img_src=cv2.imread('./example.jpg')
    img_src=cv2.imread('./dog.jpg') # BGR
    img=img_src[:,:,::-1] # RGB
    h,w,_=img.shape
    img=img.transpose(2,0,1) # [c,h,w]

    img=preprocess(img)
    img=img[None]
    img=torch.tensor(img)
    if is_cuda:
        net.cuda(did)
        img=img.cuda(did)
    boxes,labels,probs=net(img,torch.tensor([[w,h]]).type_as(img))[0]

    prob_mask=probs>cfg.out_thruth_thresh
    boxes=boxes[prob_mask ] 
    labels=labels[prob_mask ].long()
    probs=probs[prob_mask]
    draw_box(img_src,boxes,color='pred',
        text_list=[ 
            classes[_]+'[%.3f]'%(__)  for _,__ in zip(labels,probs)
            ]
        )
    show_img(img_src,-1)

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