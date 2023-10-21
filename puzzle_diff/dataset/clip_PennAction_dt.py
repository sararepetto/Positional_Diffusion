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

class PennAction_clip_dt(Dataset):
    def __init__(self,train=True, clip_len = 4, interval = 4, tuple_len = 4, subsampling=3):
        super().__init__()
        self.subsampling = subsampling
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
        self.train = train
        my_dictionary = dict()
        my_dictionary['baseball_pitch']=[]
        my_dictionary['baseball_swing']=[]
        my_dictionary['bench_press']=[]
        my_dictionary['bowl']=[]
        my_dictionary['clean_and_jerk']=[]
        my_dictionary['golf_swing']=[]
        my_dictionary['jumping_jacks']=[]
        my_dictionary['pushup']=[]
        my_dictionary['pullup']=[]
        my_dictionary['situp']=[]
        my_dictionary['squat']=[]
        my_dictionary['tennis_forehand']=[]
        my_dictionary['tennis_serve']=[]
        my_dictionary['strum_guitar']=[]
        my_dictionary['jump_rope']=[]
        video_path = os.listdir("datasets/Penn_Action/train_frames/")
        action_path = "datasets/Penn_Action/labels/"
        
        for i in range(len(video_path)):
            labels = scipy.io.loadmat(f'datasets/Penn_Action/labels/{video_path[i]}.mat')
            action = labels['action'][0]
            my_dictionary[action].append(video_path[i])
            action_list = [i for i in my_dictionary.keys()]

        self.list_files = []

        for i in range(len(action_list)):
            action=my_dictionary[action_list[i]]
            end = 0.8*len(action)
            if train == True:
                self.list_files.append(action[:int(end)])
            else:
                self.list_files.append(action[int(end):])
        self.list_files = [item for el in self.list_files for item in el]
        
        self.frames = []
        self.actions = []
        self.X_coordinates = []
        self.Y_coordinates = []
        tuples = []
        for i in range(len(self.list_files)):
            phase_path = "datasets/Penn_Action/phase_labels"
            elements = [i.replace(".npy","") for i in os.listdir(phase_path) ]
        

            if self.list_files[i] in elements:
                video_path = f"datasets/Penn_Action/train_frames/{self.list_files[i]}/*.jpg"
                phases = np.load(f"datasets/Penn_Action/phase_labels/{self.list_files[i]}.npy")
                labels = scipy.io.loadmat(f'datasets/Penn_Action/labels/{self.list_files[i]}.mat')
                self.x_coordinates = labels ['x']
                self.y_coordinates = labels['y']
                imgs = sorted(glob.glob(video_path))
                #for j in range(subsampling):
                #z = random.randint(0,self.subsampling)
                    ##[j*subsampling:(j+1)*subsampling]
                if 5 not in phases:
                    self.frames.append(imgs)
                    self.actions.append(phases)
                    self.X_coordinates.append(self.x_coordinates)
                    self.Y_coordinates.append(self.y_coordinates)
               

    def __len__(self):
        return len(self.frames)
  

    def __getitem__(self,idx):
        imgs = self.frames[idx]
        tuple_len = int((len(imgs) + 3)/7)
        actions = self.actions[idx]
        x_coordinates = self.X_coordinates[idx]
        y_coordinates = self.Y_coordinates[idx]
        tuple_clip = []
        tuple_Xcoordinates = []
        tuple_Ycoordinates = []
        action = []

        #if self.train:
            #tuple_start = random.randint(0, length - self.tuple_total_frames)
        #else:
            #random.seed(idx)
           # tuple_start = random.randint(0, length - self.tuple_total_frames)
        clip_start = 0

        for i in range(tuple_len):
           
           # breakpoint()
            #if i == self.tuple_len -1:
                #clip = imgs[clip_start:]
            #else:
            clip = imgs[clip_start: clip_start + self.clip_len]
            X_coordinates = x_coordinates[clip_start: clip_start + self.clip_len]
            Y_coordinates = y_coordinates[clip_start: clip_start + self.clip_len]
            act = actions[clip_start: clip_start + self.clip_len]
            act = torch.from_numpy(act)
            if len(clip)==4:
                action.append(act)
                tuple_clip.append(clip)
                tuple_Xcoordinates.append(X_coordinates)
                tuple_Ycoordinates.append(Y_coordinates)
            clip_start = clip_start + self.clip_len + self.interval
        videos=[]
        for i in range(len(tuple_clip)):
            #video = []
            video = []
            for j in range (self.clip_len):
        #for i in range(len(imgs)):
                image = cv2.imread(f'{tuple_clip[i][j]}')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                xmin = np.min(tuple_Xcoordinates[i][j]) - min(np.min(tuple_Xcoordinates[i][j]),20)
                xmax = np.max(tuple_Xcoordinates[i][j])+ min((image.shape[1]-np.max(tuple_Xcoordinates[i][j])),20)
                ymin = np.min(tuple_Ycoordinates[i][j]) - min(np.min(tuple_Ycoordinates[i][j]),20)
                ymax = np.max (tuple_Ycoordinates[i][j]) + min((image.shape[0]-np.max(tuple_Ycoordinates[i][j])),20)
                image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
                image = cv2.resize(image,(200,200))
                image = torch.from_numpy(image)
            #image = Image.fromarray(image.astype('uint8'), 'RGB')
                video.append(image)
            video = torch.stack(video)
            videos.append(video)
        
        video = torch.stack(videos)
        action = torch.flatten(torch.stack(action))
        return video,action
       



if __name__ == "__main__":
        dt = PennAction_clip_dt()           
        frames=0
        x= dt[20]
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0][0])
        plt.imshow(x[0][13][0])
        plt.show()                          
