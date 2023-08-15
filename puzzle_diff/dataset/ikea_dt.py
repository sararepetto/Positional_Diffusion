from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os

class Ikea_dt(Dataset):
    def __init__(self, train=True):
        super().__init__()
        data_path = Path('/media/sara/Crucial X6/Ikea/Kallax')
        if train==False:
            data = np.load("/home/sara/Project/Positional_Diffusion/datasets/Ikea/npyrecords/kallax_shelf_drawer_train.npy",allow_pickle=True)
        else:
             data = np.load("/home/sara/Project/Positional_Diffusion/datasets/Ikea/npyrecords/kallax_shelf_drawer_val.npy",allow_pickle=True)
        elements = data.tolist().keys()
        breakpoint()
        for 
        datas=os.listdir(data_path)[elements]
        breakpoint()


if __name__ == "__main__":
        dt = Ikea_dt()