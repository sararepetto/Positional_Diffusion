import argparse
import math
import os
import random
import warnings
from math import cos, radians, sin
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.loader
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class CVAE(nn.Module):
    def __init__(self, rot_heads = False):
        super().__init__()
        self.extra_pos_tokens =  0
        self.ls=256
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
        # Encoder init
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.ls, nhead=self.heads,
                                                          dim_feedforward=self.ff_size, dropout=0.1,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.layers)
        ################################################################################################################
        # Decoder init
        #seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.ls, nhead=self.heads,
                                                         # dim_feedforward=self.ff_size, dropout=0.1,
                                                         # activation=self.activation)
       # self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.layers)
       # self.finallayer = nn.Linear(self.ls, self.input_feats)
    
    def encoder(self, skeletons,batch, aug='rotate',rot_heads = False):
        self.aug = aug 
        data_perturbed = skeletons #[S*T, J, C]
        batch = batch #[S]
        x = torch.stack(
            [data_perturbed[batch == b]
             for b in batch.unique()
             ]
        )

        x  = torch.permute(x,(0,2,3,1)) #[S, J, C, T]

        bs, njoints, nfeats, nframes = x.shape
        ################################################################################################################
        # embedding of the skeleton
        x = x.permute((3, 0, 2, 1)).reshape(nframes, bs, njoints * nfeats)  # [T, S, C*J]
        x = self.skelEmbedding(x)  # [T, S, Ls]
        ################################################################################################################
        # transformer layers
        final = self.seqTransEncoder(x)  # [T, S, Ls]
        z = final.mean(axis=0)  # [S, Ls]
        enc= {'z': z,  'embedding':final}
        return enc
        
   # def decoder(self, data):
        #y= data.labels
        #z = self.encoder(data)['z'],  #data.labels # z [S Ls] -- y [S]
        #embedding=self.encoder(data)['embedding']
        # bs, latent_dim = z[0].shape
        #tgt=torch.zeros_like(embedding)
        #output = self.seqTransDecoder(tgt,memory=z[0][None])
        #da capire cosa mandare in pasto al decoder
        #output = self.finallayer(output).reshape(self.nframes, bs, self.njoints, self.nfeats)  # [T S J C]
        #output = output.permute(1, 2, 3, 0)  # [S J C T]
        #return output
    
    def forward(self, data):
        enc=self.encoder(data)
        #output = self.decoder(data)
        return enc

#puzzle_diff/dataset
if __name__ == '__main__':
    import sys 
    sys.path.append('puzzle_diff/dataset')
    from NTU_60_dt import NTU_60_dt
    from skeleton_dataset import Skeleton_dataset
    train_dt = NTU_60_dt()
    dt = Skeleton_dataset(train_dt, dataset_get_fn=lambda x: x)
    dl = torch_geometric.loader.DataLoader(dt, batch_size=2)
    dl_iter = iter(dl)
    k = next(dl_iter)
    model = DataParallel(CVAE().cuda(), device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss().cuda()
    # fname_model = '/home/sara/Project/SKELTER/model/gausNoise.pt'
    # model.load_state_dict(torch.load(fname_model), strict=False)
    device='cuda'
    output=model(k.to(device))
    breakpoint()