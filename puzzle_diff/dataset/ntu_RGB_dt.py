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

class ntu_RGB_dt(Dataset):
    def __init__(self,train=True):
        super().__init__()
        if train == False:
            self.data_path = Path("datasets/NTU_60/RGB_videos/RGB_videos_Test")
        else:
            self.data_path = Path("datasets/NTU_60/RGB_videos/RGB_videos_train")
        self.list_files=os.listdir(self.data_path)

    
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self,idx):
        
        element=self.list_files[idx]
        video_path=f"datasets/NTU_60/RGB_videos/RGB_videos_train/{element}"
        coordinate_path = f"datasets/NTU_60/coordinates/coordinates{element}.pt"
        coordinates = torch.load(coordinate_path)
        video=[]
        cap = cv2.VideoCapture(video_path)
        index = 0
        while cap.isOpened():
           Ret, Mat = cap.read()

           if Ret :#and index < 100:
               video.append(Mat)
               index += 1

           else:
               break
        cap.release()
        for i in range(len(video)):
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
            if 0 in coordinates[i]:
                video[i] = cv2.resize(video[i],(128,128))
            else:
                video[i] = video[i][coordinates[i][1]: coordinates[i][3], coordinates[i][0]:coordinates[i][2]]
                video[i] = cv2.resize(video[i],(128,128))
            video[i] = torch.from_numpy(video[i])
        video = torch.stack(video)
        video= video[::10]
        return video
    
    ##capire dove aggiungere  una funzione per calcolare il livello di movimento per ogni video 

if __name__ == "__main__":
        dt = ntu_RGB_dt()
        breakpoint()
        frames=0
        x= dt[30]
        print(len(x))
        print(x[0].shape)
        plt.figure()
        plt.imshow(x[1])
        plt.imshow(x[2])
        plt.show()