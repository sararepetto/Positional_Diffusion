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
import random

class Ikea_RGB_dt(Dataset):
    def __init__(self, train=True,subsampling=3):
        super().__init__()
        self.subsampling = subsampling
        self.elements=[]
        #sys.path.append('/datasets/Ikea/Kallax')
        data_path = Path('/datasets/Ikea/Kallax')
        #sys.path.append('/home/sara/Project/Positional_Diffusion/puzzle_diff')
        if train==True:
            data = np.load("datasets/Ikea/npyrecords/kallax_shelf_drawer_train.npy",allow_pickle=True)
        else:
             data = np.load("datasets/Ikea/npyrecords/kallax_shelf_drawer_val.npy",allow_pickle=True)
        for i in data.tolist().keys():
             self.elements.append(i)
        self.frames=[]
        self.actions=[]
        self.coordinates=[]
        dev = ['dev1','dev2','dev3']
        for i in range(len(self.elements)):
            if self.elements[i] != '.~lock.Accordo_affiliatura_09.2022_ITA_REPETTO_UNIPADOVA.docx#':
                for j in range(len(dev)):
                    sys.path.append('datasets/Ikea/Kallax')
                    video_path = f"datasets/Ikea/Kallax/{self.elements[i]}/{dev[j]}/images/*"
                    action_path = [k for k in data.tolist()[self.elements[i]]['labels']]
                    self.coordinate_path = f'datasets/Ikea/new_coordinates{self.elements[i]}{dev[j]}.pt'
                    coordinates = torch.load(self.coordinate_path)
                    imgs = sorted(glob.glob(video_path))
                    if len(imgs)== 0:
                        print(self.elements[i])
                    #for z in range(self.subsampling):
                    z = random.randint(0,self.subsampling) 
                    self.frames.append(imgs[z::self.subsampling])
                    self.actions.append(action_path[z::self.subsampling])
                    self.coordinates.append(coordinates[z::self.subsampling])
                    
    def __len__(self):
        return len(self.frames)
    
    
    def __getitem__(self,idx):
        imgs = self.frames[idx]
        actions = self.actions[idx]
        x_min = self.coordinates[idx][0][0]
        x_max = self.coordinates[idx][0][2]
        y_min = self.coordinates[idx][0][1]
        try:
            y_max = self.coordinates[idx][0][3]
        except:
            print(self.coordinates.shape)
        video=[]
        for i in range(len(imgs)):
            image = cv2.imread(f'{imgs[i]}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xmin = x_min #- min(x_min,20)
            xmax = x_max #+ min((image.shape[1]-x_max),20)
            ymin = y_min #- min(y_min,20)
            ymax = y_max #+ min((image.shape[0]-y_max),20)
            image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
            image = cv2.resize(image,(64,64))
            image = torch.from_numpy(image)
            video.append(image) 
        video = torch.stack(video)
        return video,actions
    


if __name__ == "__main__":
        dt = Ikea_RGB_dt()
        frames=0
        x= dt[10]
        print(len(x))
        breakpoint()
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0])
        plt.show()
        plt.imshow(x[0][1])
        plt.show()