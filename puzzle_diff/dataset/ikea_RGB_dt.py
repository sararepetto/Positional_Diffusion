from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
import os
import glob 
import cv2 
import torch
import sys 

class Ikea_RGB_dt(Dataset):
    def __init__(self, train=True,subsampling=3):
        super().__init__()
        self.subsampling = subsampling
        self.elements=[]
        sys.path.append('/media/sara/Crucial X6/Ikea/Kallax')
        data_path = Path('/media/sara/Crucial X6/Ikea/Kallax')
        #sys.path.append('/home/sara/Project/Positional_Diffusion/puzzle_diff')
        if train==True:
            data = np.load("datasets/Ikea/npyrecords/kallax_shelf_drawer_train.npy",allow_pickle=True)
        else:
             data = np.load("datasets/Ikea/npyrecords/kallax_shelf_drawer_val.npy",allow_pickle=True)
        for i in data.tolist().keys():
             self.elements.append(i)
        self.frames=[]
        self.actions=[]
        dev = ['dev1','dev2','dev3']
        for i in range(len(self.elements)):
            if self.elements[i] != '.~lock.Accordo_affiliatura_09.2022_ITA_REPETTO_UNIPADOVA.docx#':##come si elimina questo file?
                for j in range(len(dev)):
                    sys.path.append('/media/sara/Crucial X6/Ikea/Kallax')
                    video_path = f"/media/sara/Crucial X6/Ikea/Kallax/{self.elements[i]}/{dev[j]}/images/*"
                    action_path = [k for k in data.tolist()[self.elements[i]]['labels']]
                    imgs = sorted(glob.glob(video_path))
                    if len(imgs)== 0:
                        print(self.elements[i])
                    for z in range(self.subsampling): ##fare forse ogni 10 (chiedere a Gianluca quanto riesce a contenere come dati da riordianre
                        self.frames.append(imgs[z::self.subsampling])
                        self.actions.append(action_path[z::self.subsampling])
        breakpoint()
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,idx):
        imgs = self.frames[idx]
        actions = self.actions[idx]
        video=[]
        for i in range(len(imgs)):
            image = cv2.imread(f'{imgs[i]}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(64,64))
            image = torch.from_numpy(image)
            video.append(image) 
        video = torch.stack(video)
        return video,actions
    


if __name__ == "__main__":
        dt = Ikea_RGB_dt()
        frames=0
        x= dt[50]
        print(len(x))
        breakpoint()
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0])
        plt.show()
        plt.imshow(x[0][1])
        plt.show()