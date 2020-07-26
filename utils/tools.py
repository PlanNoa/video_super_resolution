import os
import time
import shutil
import torch
import numpy as np


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print(("  [{:.3f}{}] {}".format(duration, units, string)))

    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n" % (string))
        fid.close()


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n

    next = __next__


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def maskprocess(mask):
    return torch.stack((mask,) * 3)


def ToLabel(E):
    fgs = np.argmax(E, axis=0).astype(np.float32)
    return fgs.astype(np.uint8)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MakeCuda(x):
    return x.cuda()


def transpose1323(x):
    return x.transpose(1, 3).transpose(2, 3)


def transpose1223(x):
    return x.transpose(1, 2).transpose(2, 3)


def transpose1312(x):
    return x.transpose(1, 3).transpose(1, 2)


def transpose1201(x):
    return x.transpose(1, 2).transpose(0, 1)


def transpose030112(x):
    return x.transpose(0, 3).transpose(0, 1).transpose(1, 2)


def transpose031323(x):
    return x.transpose(0, 3).transpose(1, 3).transpose(2, 3)


import time, os, shutil, torch
import numpy as np
from collections import OrderedDict
import subprocess
import xml.etree.ElementTree

def get_gpu_usage():

    def extract(elem, tag, drop_s):
        text = elem.find(tag).text
        if drop_s not in text: raise Exception(text)
        text = text.replace(drop_s, "")
        try:
            return int(text)
        except ValueError:
            return float(text)

    d = OrderedDict()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    print('\n\nGPU utilization: %s %%\nVRAM used: %s %%\n\n' % (d["gpu_util"],d["mem_used_per"]))