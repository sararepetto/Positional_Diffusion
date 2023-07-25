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

class PennAction_RGB_dt(Dataset):
    def __init__(self,train=True):
        super().__init__()
        if train == False:
            self.data_path = Path('/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/test_frames')
        else:
            self.data_path = Path('/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/train_frames/')
        self.list_files= sorted(os.listdir(self.data_path))
    
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self,idx):
        actions=['baseball_pitch', 'clean_and_jerk','pull_ups','strumming_guitar','baseball_swing','golf_swing','push_ups','tennis_forehand' ,'bench_press','jumping_jacks','sit_ups','tennis_serve','bowling','jump_rope','squats']   
        element=self.list_files[idx]
        video_path=f"/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/train_frames/{element}"
        video=[]
        imgs = sorted(os.listdir(video_path))
        labels = scipy.io.loadmat(f'/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/labels/{element}.mat')
        x_coordinates = labels ['x']
        y_coordinates = labels['y']     
        bbox = labels['bbox']
        label = labels['action']
        action = torch.tensor(actions.index(label),dtype=torch.int8)
        breakpoint()
        for i in range(len(imgs)-1):
            image = cv2.imread(f'/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/train_frames/{element}/{imgs[i]}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xmin = np.min (x_coordinates[i]) - min(np.min(x_coordinates[i]),20)
            xmax = np.max(x_coordinates[i]) + min((image.shape[1]-np.max(x_coordinates[i])),20)
            ymin = np.min(y_coordinates[i]) - min(np.min(y_coordinates[i]),20)
            ymax = np.max (y_coordinates[i]) + min((image.shape[1]-np.max(y_coordinates[i])),20)
            X_min ,Y_min,X_max,Y_max= bbox[i]
            new_image = image[int(Y_min):int(Y_max),int(X_min):int(X_max)]
            #new_image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
            if new_image.shape[1] == 0 or new_image.shape[0] == 0 :
                image = cv2.resize(image,(64,64))
            else:
                 image = cv2.resize(new_image,(64,64))
            image = torch.from_numpy(image)
            
            video.append(image) 
        video = torch.stack(video)
        video= video[::3] 
        #video= video[::4]           
        return video,action

if __name__ == "__main__":
        dt = PennAction_RGB_dt()
        frames=0
        x= dt[30]
        breakpoint()
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[0][0])
        plt.imshow(x[0][1])
        plt.show()