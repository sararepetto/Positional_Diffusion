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
import pandas as pd
import torchvision.transforms as transforms

class UCF101_frames_dt(Dataset):
    def __init__(self,root_dir = Path('datasets/UCF101'),train=True, subsampling=3,split='1'):
        super().__init__()
        self.subsampling = subsampling
        self.train = train
        self.split = split
        self.root_dir = root_dir
        self.elements=[]
        class_idx_path = os.path.join(self.root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.train==True:
            train_split_path = os.path.join(self.root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
            self.train_split = self.train_split.tolist()
            self.train_split = [elem for elem in self.train_split]
        else:
            test_split_path = os.path.join(self.root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
            self.test_split = self.test_split.tolist()
            self.test_split = [elem for elem in self.test_split]
        print('Use split'+ self.split)
       
    def __len__(self):
            if self.train:
                return len(self.train_split)
            else:
                return len(self.test_split)
    
    def __getitem__(self, idx):
            if self.train:
                videoname = self.train_split[idx]
            else:
                videoname = self.test_split[idx]
            class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
            filename = os.path.join(self.root_dir, 'video', videoname)
            #videodata = skvideo.io.vread(filename)
            videodata=[]
            cap = cv2.VideoCapture(filename)
            index = 0
            while cap.isOpened():
                Ret, Mat = cap.read()
                #breakpoint()
                if Ret:
                    Mat1 = cv2.cvtColor(Mat, cv2.COLOR_BGR2RGB)
                    Mat1 = cv2.resize(Mat1,(126,126))
                    Mat1 = torch.from_numpy(Mat1)
                    videodata.append(Mat1)
                    index += 1
                else:
                    break
            cap.release()
            videodata = torch.stack(videodata)
            self.frames = []
            #for j in range(self.subsampling):
                #z = random.randint(0,self.subsampling)
                    ##[j*subsampling:(j+1)*subsampling]
            z = random.randint(0,self.subsampling)
            self.frames = videodata[z::self.subsampling]
                    #self.actions.append(phases[j::self.subsampling])
                    #self.X_coordinates.append(self.x_coordinates[j::self.subsampling])
                    #self.Y_coordinates.append(self.y_coordinates[j::self.subsampling]
            return self.frames,class_idx
    
if __name__ == "__main__":
        dt = UCF101_frames_dt(train=True)           
        frames=0
        x= dt[20]
        print(len(x))
        plt.figure()
        plt.imshow(x[0][0])
        #plt.imshow(np.transpose(x[0][0][0].detach().numpy(),(1,2,0)))
        plt.show()
        