import numpy as np
from os.path import *
from scipy.misc import imread
from PIL import Image
from flow_utils import readFlow

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    return []

def low_resolution(img):
    if type(img.size) == int:
        img = Image.fromarray(img)
    x, y = img.size
    img = img.resize((int(x/2), int(y/2)))
    return img