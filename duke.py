# -*- coding: utf-8 -*-
# 
# DukeMTMC-reID dataset loader
# Edited by Jianyuan Guo 
# jyguo@pku.edu.cn
# laynehuang@pku.edu.cn
# 2019.10

import sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.misc import imresize
import pdb

 
class DukeDataset(Dataset):
    def __init__(self, image_root='/home/guojianyuan/Desktop/ReId/data/DukeMTMC-reID/', txt_root=None, mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.image_root = image_root
        if self.mode == 'train':
            self.name_list = np.genfromtxt(self.image_root + 'train_list.txt', dtype=str, delimiter=' ', usecols=[0])
            self.label_list = np.genfromtxt(self.image_root + 'train_list.txt', dtype=int, delimiter=' ', usecols=[1])
        
    def __getitem__(self, index):
        img = Image.open(self.image_root + self.name_list[index])
        img = self.transform(img)
        label = self.label_list[index]
        return img, label

    def __len__(self):
        return len(self.name_list)


class DukePartDataset(Dataset):
    def __init__(self, image_root='/home/guojianyuan/Desktop/ReId/data/DukeMTMC-reID/', parsing_root="/home/huanglang/research/CE2P/duke/5classes/", mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.image_root = image_root
        self.parsing_root = parsing_root
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        self.name_list = np.genfromtxt(image_root + self.mode + '_list.txt', dtype=str, delimiter=' ', usecols=[0])
        self.label_list = np.genfromtxt(image_root + self.mode + '_list.txt', dtype=int, delimiter=' ', usecols=[1])
        
    def __getitem__(self, index):
        img = Image.open(self.image_root + self.name_list[index])
        part_map = Image.open(self.parsing_root + self.name_list[index][:-3] + "png")

        if self.mode == 'train' and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            part_map = transforms.functional.hflip(part_map)

        transforms_tensor = transforms.Compose([transforms.ToTensor()])
        img_tensor = transforms_tensor(img)
        img = self.transform(img)
        part_map = imresize(part_map, (96, 32), interp="nearest")
        part_map = torch.from_numpy(np.asarray(part_map, dtype=np.float))
        label = self.label_list[index]
        return img, label, part_map, part_map # img_tensor # For other purpose

    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    pass
