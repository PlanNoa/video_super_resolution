import cv2
from my_packages.DepthProjection.DepthProjectionModule import  DepthProjectionModule
img = cv2.imread('demo.jpg', 1)
print(type(img))
aa = DepthProjectionModule()
aa.forward(img)