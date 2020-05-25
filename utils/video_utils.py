# read video and return as frame by frame
import cv2
from torch.utils.data import Dataset
from glob import glob
import os

class VideoDataset(Dataset):
    def __init__(self, path):
        self.video_paths = glob(os.path.join(path, '*'))

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, file):
        imgs = []
        cap = cv2.VideoCapture(file)
        while cap.isopen():
            ret, img = cap.read()
            imgs.append(img)
        data = [imgs[i:i+3] for i in range(len(imgs)-2)]
        return data

    def __getitem__(self, idx):
        return self.read_video(self.video_paths[idx])