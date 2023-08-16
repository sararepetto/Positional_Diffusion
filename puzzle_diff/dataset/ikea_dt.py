from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os

class Ikea_dt(Dataset):
    def __init__(self, train=True):
        super().__init__()
        elements=[]
        data_path = Path('/media/sara/Crucial X6/Ikea/Kallax')
        if train==True:
            data = np.load("/home/sara/Project/Positional_Diffusion/datasets/Ikea/npyrecords/kallax_shelf_drawer_train.npy",allow_pickle=True)
        else:
             data = np.load("/home/sara/Project/Positional_Diffusion/datasets/Ikea/npyrecords/kallax_shelf_drawer_val.npy",allow_pickle=True)
        for i in data.tolist().keys():
             elements.append(i)
        
        breakpoint()
           # datas=data_path/i
        breakpoint()


if __name__ == "__main__":
        dt = Ikea_dt()