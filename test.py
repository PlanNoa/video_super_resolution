import cv2
import numpy as np
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule
img = cv2.imread('video_super_resolution/demo.jpg')
img = np.concatenate((img.T, img.T), 1)
aa = DepthProjectionModule()
aa.forward(img)