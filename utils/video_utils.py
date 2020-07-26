import os
import cv2
from glob import glob
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, path):
        self.video_paths = glob(os.path.join(path, '*'))
        self.data = []

    def __len__(self):
        return len(self.video_paths) * 51

    def read_video(self, file):
        imgs = []
        cap = cv2.VideoCapture(file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(length):
            ret, img = cap.read(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        data = [imgs[i:i + 3] for i in range(len(imgs) - 2)]
        for i in range(0, length, int(length/50)):
            self.data.append(data[i:i+int(length/50)])

    def __getitem__(self, idx):
        if idx % 51 == 0:
            self.read_video(self.video_paths[idx//51])
        data = self.data[0]
        self.data = self.data[1:]
        return data