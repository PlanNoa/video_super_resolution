import cv2
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule
img = cv2.imread('demo.jpg', 1)
aa = DepthProjectionModule()
aa.forward(img)