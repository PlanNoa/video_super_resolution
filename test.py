import cv2
import torch
import numpy as np
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule
img = cv2.imread('video_super_resolution/demo.jpg')
img = torch.from_numpy(img)

img = img.transpose(0, 2)

img = img.transpose(1, 2)
print(img.size())
img = torch.cat([img, img], dim=0)
img = torch.stack([img, img], dim=0)

print(img.size())
aa = DepthProjectionModule()
aa.forward(img)