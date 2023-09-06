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

class PennAction_RGB_dt(Dataset):
    def __init__(self,train=True, subsampling=3):
        super().__init__()
        self.subsampling = subsampling
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
        #numpy.randompermutation
        #seed
        
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
                for j in range(subsampling):
                #z = random.randint(0,self.subsampling)
                    self.frames.append(imgs[j::self.subsampling])
                    self.actions.append(phases[j::self.subsampling])
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
            image = cv2.resize(image,(70,70))
            image = torch.from_numpy(image)
            #image = Image.fromarray(image.astype('uint8'), 'RGB')
            video.append(image) 
        video = torch.stack(video)
        return video,actions




if __name__ == "__main__":
        dt = PennAction_RGB_dt()           
        frames=0
        x= dt[20]
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0])
        plt.imshow(x[0][1])
        plt.show()                          
