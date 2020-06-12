import cv2
import random
import torch
import numpy as np
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule

input_frame_size = (3, 256, 448)
output_frame_size = (3, 256, 448)

h_offset = random.choice(range(256 - input_frame_size[1] + 1))
w_offset = random.choice(range(448 - input_frame_size[2] + 1))


img = cv2.imread('video_super_resolution/demo.jpg')
img = img[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
print(np.shape(img))

img = np.transpose(img, (2,0,1))
img = img.astype("float32")/ 255.0
from PIL import Image

img = Image.fromarray(img)
img = img.resize((256, 448))
img = torch.from_numpy(img)
#img = torch.cat([img, img], dim=0)
img = torch.stack([img, img], dim=0)
print(img.size())
aa = DepthProjectionModule()
aa.forward(img)