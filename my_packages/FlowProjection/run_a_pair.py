import torch
import numpy as np
import argparse
from utils.frame_utils import read_gen
import time
from utils.flow_utils import *
from models import FlowNet2

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

def visulize_flow(flow):
    flow_data = flow
    img = flow2img(flow_data)
    plt.imsave("result.png", img)

now = time.time()
net = FlowNet2().cuda()
dict = torch.load("pretrained/FlowNet2_checkpoint.pth.tar")
net.load_state_dict(dict["state_dict"])

# load the image pair, you can find this operation in dataset.py
img1 = read_gen("data/00000"+".jpg")
img2 = read_gen("data/00001"+".jpg")
images = [img1, img2]
print(img1.shape)
image_size = img1.shape[:2]

frame_size = img1.shape
render_size = [((frame_size[0])//64 )*64, ((frame_size[1])//64)*64]

cropper = StaticCenterCrop(image_size, render_size)
images = list(map(cropper, images))

images = np.array(images).transpose(3,0,1,2)
images = torch.from_numpy(images.astype(np.float32))
print(images.shape)
images = images.unsqueeze(0).cuda()

        # process the image pair to obtian the flow
result = net(images).squeeze()
print(images.shape)
print(result.shape)

flow = result.data.cpu().numpy().transpose(1, 2, 0)
visulize_flow(flow)
print(flow.shape)