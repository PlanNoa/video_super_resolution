import cv2
import numpy as np
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule
img = cv2.imread('video_super_resolution/demo.jpg')
print(img.size())
img = img.transpose(0, 2)

img = img.transpose(1, 2)
print(img.size())
img = np.column_stack((img, img), 1)
aa = DepthProjectionModule()
aa.forward(img)