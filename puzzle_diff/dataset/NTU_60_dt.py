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

def randominit():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(args.seed)


def log(s, nl=True):
    print(s, end='\n' if nl else '')



def load_data():
    train_data = torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/train_data_1.pt')
    train_label = torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/train_label_1.pt')
    test_data = torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/test_data_1.pt')
    test_label = torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/test_label_1.pt')
    train_dataset = SkeletonDataset(train_data, train_label, aug=args.aug)
    test_dataset = SkeletonDataset(test_data, test_label, aug=args.aug)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0,
                              pin_memory=True, collate_fn=train_dataset.collate)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=test_dataset.collate)
    return train_loader, test_loader


########################################################################################################################
class SkeletonDataset(Dataset):
    def __init__(self, data, label, aug=None,rot_angle=0):
        super().__init__()
        self.aug = aug
        self.rot_angle= rot_angle
        self.transform = aug_transform()
        self.data = torch.stack(data)
        self.label = torch.stack(label)
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
                rot = self.convert_rot_labels(args.rot_angle)
        label = self.label[index].item()
        return data, label, rot, data_aug  # [J, C, T]

    @staticmethod
    def collate(batch):
        databatchTensor = torch.stack([b[0] for b in batch], dim=0)  # [S, J, C, T]
        labelbatchTensor = torch.as_tensor([b[1] for b in batch])  # [S]
        rotbatchTensor = torch.as_tensor([b[2] for b in batch])
        augbatchTensor = torch.stack([b[3] for b in batch], dim=0)  # [S, J, C, T]
        batch = {'x': databatchTensor,  # 4d tensor [bs, joints, chans, frames]
                 'y': labelbatchTensor,  # labels of each batch sample
                 'rot': rotbatchTensor,
                 'aug': augbatchTensor}
        return batch

    def reverse(self, index):
        return self.data[index].permute(1, 0, 2).float().clone()[:, :, list(reversed([i for i in range(self.T)]))]

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

    # Training settings
    parser.add_argument('--seed', type=int, default=1990, help='random seed (default: 1990)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)')

    # I/O settings
    parser.add_argument('--save_data', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='/media/gpaoletti/2TB/SKELTER/ECCV/saved_files/model')
    parser.add_argument('--data_path', type=str, default='/home/sara/Project/SKELTER/NTU')

    # Model settings
    parser.add_argument('--pos_dropout', type=float, default=0.1, help='position embedding dropout rate (default: 0.1')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads (default: 4)')
    parser.add_argument('--layers', type=int, default=2, help='number of transformer blocks (default: 8)')
    parser.add_argument('--ls', type=int, default=256, help='latent space size (default: 256)')
    parser.add_argument('--max_len', type=int, default=100, help='maximum length of position embedding (default: 5000')
    parser.add_argument('--mlp_dropout', type=float, default=0.1)
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[256, 256, 128])
    parser.add_argument('--mlp_lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')

    # Ablation settings
    parser.add_argument('--split', type=str, default='xview60', choices=['xsub60', 'xview60', 'xsub120', 'xview120'])
    parser.add_argument('--aug', type=str, default='gausNoise',
                        help="choose a single or combos (_-separated) of "
                             " 'gausNoise', 'shear', 'subtract', 'zeroOutAxis', 'zeroOutJoints', 'zeroOutLimbs' "
                             " 'outlier', 'rotate_all', 'rotate_rand', 'contrastive', 'None' ")

    parser.add_argument('--distribution', type=bool, default=True)

    args = parser.parse_args()
    args.nfeats = 3
    args.njoints = 25
    args.nframes = 100

    args.rot_heads = True if 'rotate' in args.aug else False
    args.contrastive = True if args.aug == 'contrastive' else False
    args.aug = 'None' if args.aug == 'contrastive' else args.aug

    return args


if __name__ == '__main__':
    args = parseargs()
    randominit()
    train_loader, test_loader = load_data()
    el= next(iter(train_loader))
    breakpoint()