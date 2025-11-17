import os
import json
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs
from PIL import Image


class DigitsDataset(data.Dataset):
    def __init__(self, path, train=True, transform_func=None):
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transform = transform_func

        with open(os.path.join(path, 'format.json'), 'r') as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, ind):
        img_path, target = self.files[ind]
        t = self.targets[target]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img).ravel().float() / 255.0
        return img, t
    
    def __len__(self):
        return self.length
    

