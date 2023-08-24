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

class PennAction_RGB_dt(Dataset):
    def __init__(self,train=True, subsampling=3):
        super().__init__()
        self.subsampling = subsampling
        my_dictionary = dict()
        my_dictionary['baseball_pitch']=[]
        my_dictionary['baseball_swing']=[]
        my_dictionary['bench_press']=[]
        my_dictionary['bowling']=[]
        my_dictionary['clean_and_jerk']=[]
        my_dictionary['golf_swing']=[]
        my_dictionary['jumping_jacks']=[]
        my_dictionary['pushups']=[]
        my_dictionary['pullups']=[]
        my_dictionary['situp']=[]
        my_dictionary['squats']=[]
        my_dictionary['tennis_forehand']=[]
        my_dictionary['tennis_serve']=[]
       
        if train==False:
            self.data_path=[]
            self.phase=[]
            for i in my_dictionary.keys():
                pos=sorted(os.listdir(f"datasets/Penn_Action/penn_action_labels/val/{i}"))
                new_pos = [sub.replace('.npy','') for sub in pos]
                self.data_path.append(new_pos)
                for j in range(len(pos)):
                    phase = np.load(f"datasets/Penn_Action/penn_action_labels/val/{i}/{pos[j]}")
                    self.phase.append(phase)
            self.data=[element for action in self.data_path for element in action]
            

        else:
            self.data_path=[]
            self.phase=[]
            for i in my_dictionary.keys():
                pos=sorted(os.listdir(f"datasets/Penn_Action/penn_action_labels/train/{i}"))
                new_pos= [sub.replace('.npy','') for sub in pos]
                self.data_path.append(new_pos)
                for j in range(len(pos)):
                    phase = np.load(f"datasets/Penn_Action/penn_action_labels/train/{i}/{pos[j]}")
                    self.phase.append(phase)
            self.data=[element for action in self.data_path for element in action]
        self.list_files= self.data
    
        self.frames=[]
        self.actions=[]
        self.X_coordinates=[]
        self.Y_coordinates=[]

        for i in range(len(self.list_files)):
            video_path = f"datasets/Penn_Action/train_frames/{self.list_files[i]}/*.jpg"
            action_path = self.phase[i]
            labels = scipy.io.loadmat(f'datasets/Penn_Action/labels/{self.list_files[i]}.mat')
            self.x_coordinates = labels ['x']
            self.y_coordinates = labels['y']
            imgs = sorted(glob.glob(video_path))
            for j in range(subsampling):
                #z = random.randint(0,self.subsampling)
                self.frames.append(imgs[j::self.subsampling])
                self.actions.append(action_path[j::self.subsampling])
                self.X_coordinates.append(self.x_coordinates[j::self.subsampling])
                self.Y_coordinates.append(self.y_coordinates[j::self.subsampling])

    def __len__(self):
        return len(self.frames)
  

    def __getitem__(self,idx):
        imgs = self.frames[idx]
        actions = self.actions[idx]
        x_coordinates = self.X_coordinates[idx]
        y_coordinates = self.Y_coordinates[idx]
        video=[]
        for i in range(len(imgs)):
            image = cv2.imread(f'{imgs[i]}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xmin = np.min(x_coordinates[i]) - min(np.min(x_coordinates[i]),20)
            xmax = np.max(x_coordinates[i])+ min((image.shape[1]-np.max(x_coordinates[i])),20)
            ymin = np.min(y_coordinates[i]) - min(np.min(y_coordinates[i]),20)
            ymax = np.max (y_coordinates[i]) + min((image.shape[0]-np.max(y_coordinates[i])),20)
            image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
            image = cv2.resize(image,(64,64))
            image = torch.from_numpy(image)
            video.append(image) 
        video = torch.stack(video)
        return video,actions

if __name__ == "__main__":
        dt = PennAction_RGB_dt()
        frames=0
        x= dt[30]
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0])
        plt.imshow(x[0][1])
        plt.show()
