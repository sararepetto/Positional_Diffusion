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

class UCF101_clip_dt(Dataset):
    def __init__(self,root_dir = Path('datasets/UCF101'),train=True, clip_len =16, interval = 8, tuple_len = 3, subsampling=3,split='1'):
        super().__init__()
        self.subsampling = subsampling
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
        self.train = train
        self.split = split
        self.root_dir = root_dir
        self.elements=[]
        self.to_remove = torch.load('datasets/UCF101/videos.pt')
        class_idx_path = os.path.join(self.root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.train==True:
            train_split_path = os.path.join(self.root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
            self.train_split = self.train_split.tolist()
            self.train_split = [elem for elem in self.train_split if elem not in self.to_remove]
        else:
            test_split_path = os.path.join(self.root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
            self.test_split = self.test_split.tolist()
            self.test_split = [elem for elem in self.test_split if elem not in self.to_remove]
        #breakpoint()
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
                if Ret:
                    videodata.append(Mat)
                    index += 1
                else:
                    break
            cap.release()
            videodata = np.array(videodata)
            length, height, width, channel = videodata.shape
            tuple_clip = []
            all_idx = []
            if self.train:
                self.transform = transforms.Compose(
                    [    
                         transforms.ToPILImage(),
                         transforms.Resize((128, 171)),
                         transforms.RandomCrop(112),
                         transforms.ToTensor(),
            
                    ]
                )
            else:
                 self.transform = transforms.Compose(
                    [    transforms.ToPILImage(),
                         transforms.Resize((128, 171)),
                         transforms.CenterCrop(112),
                         transforms.ToTensor(),
            
                    ]
                 )
            tuple_order = list(range(0, self.tuple_len))
            if self.train:
                tuple_start = random.randint(0, length - self.tuple_total_frames)
            else:
                random.seed(idx)
                tuple_start = random.randint(0, length - self.tuple_total_frames)
            # random select a clip for train
            clip_start = tuple_start
            for _ in range(self.tuple_len):
                trans_clip = []
                clip = videodata[clip_start: clip_start + self.clip_len]
                seed = random.random()
                for frame in clip:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.transform(frame)
                    trans_clip.append(frame)
                clip = torch.stack(trans_clip)
                tuple_clip.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))
                clip_start = clip_start + self.clip_len + self.interval
            tuple_clip = torch.stack(tuple_clip).permute(0,1,3,4,2)
            return tuple_clip,all_idx
           
       
            
        

if __name__ == "__main__":
        dt = UCF101_clip_dt()           
        frames=0
        x= dt[20]
        print(len(x))
        plt.figure()
        plt.imshow(x[0][0][0])
        #plt.imshow(np.transpose(x[0][0][0].detach().numpy(),(1,2,0)))
        plt.show()
        