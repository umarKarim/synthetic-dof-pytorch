import torch
from torchvision import transforms  
import cv2 
import urllib.request 
import matplotlib.pyplot as plt 
from options import Options 

class DepthExtractor():
    def __init__(self, opts):
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.model.eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", 'transforms').default_transform
        self.save_depth = opts.save_depth 
        self.depth_dir = opts.depth_dir 

    def __call__(self, im):
        net_in = self.transforms(im) 
        with torch.no_grad():
            net_out = self.model(net_in)
            net_out_res = torch.nn.functional.interpolate(net_out.unsqueeze(1), 
                                                          im.shape[:2],
                                                          mode='bicubic',
                                                          align_corners=False
                                                          ).squeeze()
        dmap = net_out_res.numpy()
        if self.save_depth:
            dmap_temp = (dmap.max() - dmap) / (dmap.max() - dmap.min())
            cv2.imwrite(self.depth_dir + 'dmap.png', dmap_temp * 255)
        return dmap



        



