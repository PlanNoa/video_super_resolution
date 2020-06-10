import cv2
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule
img = cv2.imread('video_super_resolution/demo.jpg')
aa = DepthProjectionModule()
aa.forward(img)