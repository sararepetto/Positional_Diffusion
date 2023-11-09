import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

class Diffusion_embedding(Dataset):
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embs,label = self.data[idx]
        #breakpoint()
        #if type(label)=='list':
        label = torch.cat([torch.from_numpy(x) for x in label])
        #label = torch.tensor(label)[0]
        #else:
            #label = torch.from_numpy(label)
        label = torch.unsqueeze(label,dim =-1)
        return embs,label