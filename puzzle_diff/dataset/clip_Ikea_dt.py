from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import torch
import skvideo.io
from skimage.transform import resize
import scipy.io
import glob
import random
from PIL import Image
import string
import sys

class IKEA_clip_dt(Dataset):
    def __init__(self,train=True, clip_len =16, interval = 16, tuple_len = 3, subsampling=3):
        super().__init__()
        self.subsampling = subsampling
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
        self.train = train
        self.elements=[]
        data_path = Path('/datasets/Ikea/Kallax')
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
                    #self.coordinate_path = f'datasets/Ikea/new_coordinates{self.elements[i]}{dev[j]}.pt'
                    #coordinates = torch.load(self.coordinate_path)
                    imgs = sorted(glob.glob(video_path))
                    if len(imgs)== 0:
                        print(self.elements[i])
                    self.frames.append(imgs)
                    self.actions.append(action_path)
                    #self.coordinates.append(coordinates)


    def __len__(self):
        return len(self.frames)
  

    def __getitem__(self,idx):
        imgs = self.frames[idx]
        actions = self.actions[idx]
        #x_coordinates = self.X_coordinates[idx]
        #y_coordinates = self.Y_coordinates[idx]
        length = len(imgs)
        tuple_clip = []
        #tuple_Xcoordinates = []
        #tuple_Ycoordinates = []

        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        clip_start = tuple_start

        for _ in range(self.tuple_len):
            clip = imgs[clip_start: clip_start + self.clip_len]
            #X_coordinates = x_coordinates[clip_start: clip_start + self.clip_len]
            #Y_coordinates = y_coordinates[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            #tuple_Xcoordinates.append(X_coordinates)
            #tuple_Ycoordinates.append(Y_coordinates)
            #clip_start = clip_start + self.clip_len + self.interval

        videos=[]
        for i in range(len(tuple_clip)):
            #video = []
            video = []
            for j in range (self.clip_len):
        #for i in range(len(imgs)):
                image = cv2.imread(f'{tuple_clip[i][j]}')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #xmin = np.min(tuple_Xcoordinates[i][j]) - min(np.min(tuple_Xcoordinates[i][j]),20)
                #xmax = np.max(tuple_Xcoordinates[i][j])+ min((image.shape[1]-np.max(tuple_Xcoordinates[i][j])),20)
                #ymin = np.min(tuple_Ycoordinates[i][j]) - min(np.min(tuple_Ycoordinates[i][j]),20)
                #ymax = np.max (tuple_Ycoordinates[i][j]) + min((image.shape[0]-np.max(tuple_Ycoordinates[i][j])),20)
                #image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
                image = cv2.resize(image,(150,150))
                image = torch.from_numpy(image)
            #image = Image.fromarray(image.astype('uint8'), 'RGB')
                video.append(image)
            video = torch.stack(video)
            videos.append(video)
        
        video = torch.stack(videos)
        return video,actions      

if __name__ == "__main__":
        dt = IKEA_clip_dt()           
        frames=0
        x= dt[20]
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0][0])
        plt.imshow(x[0][1][0])
        plt.show() 