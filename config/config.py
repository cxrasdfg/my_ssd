# coding=UTF-8

class CFG():
    voc_dir='/home/theppsh/workspace/data/voc/VOCdevkit/VOC2007'

    device_id=0
    use_cuda=True

    weights_dir='./weights/'

    caffe_model="./models/vgg16_caffe_pretrain.pth"
    
    # loc_mean=[.0,.0,.0,.0]
    # loc_std=[1.,1.,2.,2.]

    use_batchnorm=False

    intput_w=300
    input_h=300

    smin=.2
    smax=.9

    l2norm_scale=20

    rand_seed=0
    
    epochs=10
    lr=1e-3
    weight_decay=0.0005
    use_adam=False

    out_thruth_thresh=.5
    pos_thresh=.5

    def _print(self):
        print('Config')
        print('{')
        for k in self._attr_list():
            print('%s=%s'% (k,getattr(self,k)) )
        print("}")
    @staticmethod

    def _attr_list():
        return [k for k in CFG.__dict__.keys() if not k.startswith('_') ] 
            
    # rpn path
   
cfg=CFG()