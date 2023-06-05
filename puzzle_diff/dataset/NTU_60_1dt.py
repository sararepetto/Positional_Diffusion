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

def randominit(seed=1990):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(seed)

class NTU_60_dt(Dataset):
    def __init__(self,aug=None,rot_angle=0):
        super().__init__()
        self.aug = aug
        self.rot_angle= rot_angle
        self.transform = aug_transform()
        self.data= torch.stack(torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/train_data_1.pt'))
        self.label=torch.stack(torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/train_label_1.pt'))
        self.N, _, _, self.T = self.data.shape
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data = self.data[index].permute(1, 0, 2).float().clone()  # [J, C, T]
        data_aug = torch.empty([])
        rot = torch.empty([])
        if self.aug != 'None':
            data_aug = self.transform(data.permute(1, 2, 0).unsqueeze(-1).numpy()).squeeze().permute(2, 0, 1).float()
            if 'rotate' in args.aug:
                rot = self.convert_rot_labels(self.rot_angle)
        label = self.label[index].item()
        return data, label, rot, data_aug  # [J, C, T]

    @staticmethod
    def convert_rot_labels(rot):
        if rot[0] < 0:
            rot[0] = abs(rot[0]) + 30
        if rot[1] < 0:
            rot[1] = abs(rot[1]) + 30
        return rot



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



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, default='gausNoise',
                        help="choose a single or combos (_-separated) of "
                             " 'gausNoise', 'shear', 'subtract', 'zeroOutAxis', 'zeroOutJoints', 'zeroOutLimbs' "
                             " 'outlier', 'rotate_all', 'rotate_rand', 'contrastive', 'None' ")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    randominit()
    dt = NTU_60_dt()
    x = dt[30]
    breakpoint()