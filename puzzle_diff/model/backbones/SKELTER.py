import argparse
import math
import os
import random
import warnings
from math import sin, cos, radians
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
import torch_geometric.loader


class NTU_60_dt1(Dataset):
    def __init__(self,aug='rotate',train=True):
        super().__init__()
        self.aug = aug
        self.transform = aug_transform()
        self.angle_y= random.randint(-30, 30)
        beta = radians(self.angle_y)
        self.rot_angle = [0, self.angle_y, 0]
        if train == False:
            self.data= torch.stack(torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/test_data_1.pt'))
            self.label=torch.stack(torch.load('/home/sara/Project/SKELTER/NTU/NTU_60/xview/test_label_1.pt'))  
        else:
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
            if 'rotate' in self.aug:
                rot = self.convert_rot_labels(self.rot_angle)
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
########################################################################################################################
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.ls=256
        self.dropout = nn.Dropout(p=0.1)
        self.layer = []
        self.mlp_layer=[256, 256, 128]
        self.num_classes=20
        if not len(self.mlp_layer) == 1:
            for i in range(len(self.mlp.layer)):
                io = [self.ls, [i]] if i == 0 else [self.mlp_layers[i - 1], self.mlp_layers[i]]
                self.layer.append(nn.Linear(io[0], io[1]))
        self.layer.append(nn.Linear(self.mlp_layers[-1], self.num_classes))
        self.layer = nn.ModuleList(self.layer)

    def forward(self, x):
        if not len(self.mlp_layers) == 1:
            for i in range(len(self.mlp_layers)):
                x = self.relu(self.dropout(self.layer[i](x)))
        x = self.layer[-1](x)
        return x


########################################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, extra_pos_tokens=0):
        super(PositionalEncoding, self).__init__()
        self.ls=256
        self.max_len=100
        self.extra_pos_tokens = extra_pos_tokens
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(self.max_len, self.ls)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.ls, 2).float() * (-np.log(10000.0) / self.ls))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # [args.max_len, 1, args.ls]
        if extra_pos_tokens != 0:
            self.pe_rot = nn.Parameter(torch.zeros(extra_pos_tokens, self.ls).unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x[self.extra_pos_tokens:] = x[self.extra_pos_tokens:] + self.pe
        if self.extra_pos_tokens != 0:
            x[:self.extra_pos_tokens] = x[:self.extra_pos_tokens] + self.pe_rot
        return self.dropout(x)


class CVAE(nn.Module):
    def __init__(self, rot_heads = False):
        super().__init__()
        ################################################################################################################
        # Rotation heads
        self.extra_pos_tokens = 3 if rot_heads==True else 0
        self.ls=256
        if rot_heads==True:
            self.pre_logits_rot = nn.Identity()
            self.rot_head_x = nn.Linear(self.ls, 61)
            self.rot_head_y = nn.Linear(self.ls, 61)
            self.rot_head_z = nn.Linear(self.ls, 361)
            self.rot_token_x = nn.Parameter(torch.zeros(1, 1, self.ls))
            self.rot_token_y = nn.Parameter(torch.zeros(1, 1, self.ls))
            self.rot_token_z = nn.Parameter(torch.zeros(1, 1, self.ls))
        self.heads=4
        self.njoints=25
        self.nfeats=3
        self.nframes = 100
        self.ff_size = self.ls * self.heads
        self.activation = 'gelu'
        self.input_feats = self.njoints * self.nfeats
        self.layers=8
        ################################################################################################################
        # Skeleton embedding
        self.skelEmbedding = nn.Linear(self.input_feats, self.ls)
        ################################################################################################################
        # Positional embedding
        self.sequence_pos_encoder = PositionalEncoding(self.extra_pos_tokens)
        ################################################################################################################
        # Encoder init
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.ls, nhead=self.heads,
                                                          dim_feedforward=self.ff_size, dropout=0.1,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.layers)
        ################################################################################################################
        # Decoder init
        self.sequence_pos_decoder = PositionalEncoding()
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.ls, nhead=self.heads,
                                                          dim_feedforward=self.ff_size, dropout=0.1,
                                                          activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.layers)
        self.finallayer = nn.Linear(self.ls, self.input_feats)
    
    def encoder(self, data,aug='rotate',rot_heads = False):
        self.aug = aug 
        data_perturbed=data.perturbed #[S*T, J, C]
        batch = data.batch #[S]
        x_1 = data_perturbed[batch == 0]
        x_2 = data_perturbed[batch == 1]
        x= torch.stack((x_1,x_2)) #[S, T, J, C]
        x= torch.permute(x,(0,2,3,1)) #[S, J, C, T]

        bs, njoints, nfeats, nframes = x.shape
        ################################################################################################################
        # embedding of the skeleton
        x = x.permute((3, 0, 2, 1)).reshape(nframes, bs, njoints * nfeats)  # [T, S, C*J]
        x = self.skelEmbedding(x)  # [T, S, Ls]
        ################################################################################################################
        # add positional encoding
        if rot_heads == True:
            rot_token_x = self.rot_token_x.expand(-1, x.shape[1], -1)  # [1 1 Ls] -> [1 S Ls]
            rot_token_y = self.rot_token_y.expand(-1, x.shape[1], -1)  # [1 1 Ls] -> [1 S Ls]
            rot_token_z = self.rot_token_z.expand(-1, x.shape[1], -1)  # [1 1 Ls] -> [1 S Ls]
            x = torch.cat((rot_token_x, rot_token_y, rot_token_z, x), dim=0)  # stack rot token on top of pos_embed
        x = self.sequence_pos_encoder(x)  # [T, S, Ls]
        ################################################################################################################
        # transformer layers
        final = self.seqTransEncoder(x)  # [T, S, Ls]
        x_rot = None
        if rot_heads == True:
            rot_x = self.pre_logits_rot(final[0, :])  # [S Ls]
            rot_y = self.pre_logits_rot(final[1, :])  # [S Ls]
            rot_z = self.pre_logits_rot(final[2, :])  # [S Ls]
            final = final[self.extra_pos_tokens:, :]  # [T S Ls]
            x_rot = [self.rot_head_x(rot_x), self.rot_head_y(rot_y), self.rot_head_z(rot_z)]  # [S Rots]
        # get the average of the output
        z = final.mean(axis=0)  # [S, Ls]
        enc= {'z': z, 'rot_hat': x_rot, 'embedding':final}
        return enc
        
    def decoder(self, data):
        y= data.labels
        z = self.encoder(data)['z'],  #data.labels # z [S Ls] -- y [S] 
        bs, latent_dim = z[0].shape
        timequeries = self.sequence_pos_decoder(torch.zeros(self.nframes, bs, latent_dim, device='cuda'))  # T S Ls
        output = self.seqTransDecoder(tgt=timequeries, memory=z[0][None])  # [T S Ls]
        output = self.finallayer(output).reshape(self.nframes, bs, self.njoints, self.nfeats)  # [T S J C]
        output = output.permute(1, 2, 3, 0)  # [S J C T]
        return output
    
    def forward(self, data):
        enc=self.encoder(data)
        output = self.decoder(data)
        return enc,output


from pathlib import Path

#/home/sara/Project/Positional_Diffusion/puzzle_diff/dataset
if __name__ == '__main__':
    import sys 
    sys.path.append('/home/sara/Project/Positional_Diffusion/puzzle_diff/dataset')
    from NTU_60_1dt import NTU_60_dt
    from skeleton_dataset import Skeleton_dataset
    train_dt = NTU_60_dt()
    dt = Skeleton_dataset(train_dt, dataset_get_fn=lambda x: x)
    dl = torch_geometric.loader.DataLoader(dt, batch_size=2)
    dl_iter = iter(dl)
    k = next(dl_iter)
    model = DataParallel(CVAE().cuda(), device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss().cuda()
    fname_model = '/home/sara/Project/SKELTER/model/gausNoise.pt'
    model.load_state_dict(torch.load(fname_model), strict=False)
    device='cuda'
    output=model(k.to(device))
    breakpoint()
    
    