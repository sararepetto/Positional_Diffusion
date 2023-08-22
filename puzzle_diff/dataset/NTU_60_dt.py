import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class NTU_60_dt(Dataset):
    def __init__(self,train=True):
        super().__init__()
        self.transform = aug_transform()
        if train == False:
            self.data= torch.stack(torch.load('datasets/NTU_60/xsub/test_data_1.pt'))
            self.label=torch.stack(torch.load('datasets/NTU_60/xsub/test_label_1.pt'))  
        else:
            self.data= torch.stack(torch.load('datasets/NTU_60/xsub/train_data_1.pt'))
            self.label=torch.stack(torch.load('datasets/NTU_60/xsub/train_label_1.pt'))
        self.N, _, _, self.T = self.data.shape
        self.new_data=[]
        
        for i in range(self.N):
            for j in range(6):
                self.new_data.append(self.data[i][:,:,j::6].permute(2,0,1))


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.new_data[index]
        data = self.new_data[index]# [J, C, T]
        #data_aug = self.transform((data).permute(1, 2, 0).unsqueeze(-1).numpy()).squeeze().permute(2, 0, 1).float()
        #label = self.label[index].item()
        return data# [J, C, T]

    



def aug_look():
        return Gaus_noise()
  
class  Gaus_noise(object):
    def __init__(self, mean=0, std=0.05):
        self.mean = 0 if mean is None else mean
        self.std = 0.05 if std is None else std

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        return temp + noise

class ToTensor(object):
    def __call__(self, data_numpy):
        return torch.from_numpy(data_numpy)


def aug_transform():
    transform_aug = []
    augmentation = aug_look()
    transform_aug.append(augmentation)
    transform_aug.extend([ToTensor(), ])
    transform_aug = Compose(transform_aug)
    return transform_aug



if __name__ == '__main__':
    dt = NTU_60_dt()
    x = dt[30]
    
