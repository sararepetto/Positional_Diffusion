from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import torch


class ntu_POSE_dt(Dataset):
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
        video=[]
        cap = cv2.VideoCapture(video_path)
        index = 0
        while cap.isOpened():
           Ret, Mat = cap.read()

           if Ret and index < 100:
                video.append(Mat)
                index += 1

           else:
                break
        cap.release()
        for i in range(len(video)):
           video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
           #con open per forza usare l'object detection per avere delle rsiposte adeguate
           #video[i] = cv2.resize(video[i],(int((video[i].shape[1])/8), int((video[i].shape[0])/8)))
           #video[i] = cv2.resize(video[i], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
           #imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue) #(184,328,3)
           video[i] = cv2.resize(video[i],(int((video[i].shape[1])/8), int((video[i].shape[0])/8)))
           video[i] = np.transpose(np.float32(video[i]), (2, 0, 1)) / 256 - 0.5 
                #im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
           video[i] = np.ascontiguousarray(video[i])
           video[i] = torch.from_numpy(video[i])
        video = torch.stack(video)
        video= video[::10]
           #video = video[::10]
           #video[i] = torch.from_numpy(video[i])
           #video[i] = cv2.resize(video[i],(64,64))
           #video[i] = cv2.resize(video[i],(int((video[i].shape[1])/24), int((video[i].shape[0])/24)))
           #video[i] = cv2.resize(video[i],(int((video[i].shape[1])/8), int((video[i].shape[0])/8)))

        return video

if __name__ == "__main__":
        dt = ntu_POSE_dt()
        breakpoint()
        frames=0
        x= dt[30]
        print(len(x))
        plt.figure()
        plt.imshow(np.transpose(x[2],(1,2,0)))
        plt.show()