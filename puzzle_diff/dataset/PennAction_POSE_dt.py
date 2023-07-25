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

class PennAction_POSE_dt(Dataset):
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
        element=self.list_files[idx]
        video_path=f"/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/train_frames/{element}"
        video=[]
        imgs = sorted(os.listdir(video_path))
        labels = scipy.io.loadmat(f'/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/labels/{element}.mat')
        x_coordinates = labels ['x']
        y_coordinates = labels['y']
        bbox = labels['bbox']
        for i in range(len(imgs)):
            image = cv2.imread(f'/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/train_frames/{element}/{imgs[i]}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xmin = np.min(x_coordinates[i]) - min(np.min(x_coordinates[i]),20)
            xmax = np.max(x_coordinates[i])+ min((image.shape[1]-np.max(x_coordinates[i])),20)
            ymin = np.min(y_coordinates[i]) - min(np.min(y_coordinates[i]),20)
            ymax = np.max (y_coordinates[i]) + min((image.shape[1]-np.max(x_coordinates[i])),20)
            #image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
            #X_min ,Y_min,X_max,Y_max= bbox[i]
            image = cv2.resize(image,(128,128))#provare a togliere il resize e prendere soltanto l'embedding 18
            image = np.float32(image) / 256 - 0.5 
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image)
            video.append(image)
            
        video = torch.stack(video)
        video= video[::4] #prima5
        return video

if __name__ == "__main__":
        dt = PennAction_POSE_dt()
        frames=0
        x= dt[30]
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[1])
        plt.imshow(x[2])
        plt.show()
     