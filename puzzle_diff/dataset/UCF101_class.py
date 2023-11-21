import os
from pathlib import Path
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
import cv2
import numpy as np
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

class UCF101Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir = Path('datasets/UCF101'), clip_len =16, split='1', train=True, transforms_=None, test_sample_num=10):
        super().__init__()
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)
    
    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)
    
    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
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
        
        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]
            self.transform = transforms.Compose(
                    [    
                         transforms.ToPILImage(),
                         transforms.Resize((128, 171)),
                         transforms.RandomCrop(112),
                         transforms.ToTensor(),
            
                    ]
                )
            trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = self.transform(frame) # tensor [C x H x W]
                trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
            clip = torch.unsqueeze(torch.stack(trans_clip),dim=0)
            clip = clip.permute([0,1, 3, 4, 2])
            return clip, torch.tensor(int(class_idx)-1)
    

        else:
            self.transform = transforms.Compose(
                    [    transforms.ToPILImage(),
                         transforms.Resize((128, 171)),
                         transforms.CenterCrop(112),
                         transforms.ToTensor(),
            
                    ]
                 )
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                clip = videodata[clip_start: clip_start + self.clip_len]
                trans_clip = [] 
                seed = random.random()
                for frame in clip:
                        random.seed(seed)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = self.transform(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([0, 2, 3, 1])
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))
            return torch.stack(all_clips), torch.tensor(int(class_idx)-1)
        

if __name__ == "__main__":
        dt = UCF101Dataset(train = False)           
        frames=0
        x= dt[20]
        breakpoint()
        frames=x[0].permute(0,2,3,4,1)
        print(len(x))
        plt.figure()
        plt.imshow(frames[0][0])
        #plt.imshow(np.transpose(x[0][0][0].detach().numpy(),(1,2,0)))
        plt.show()

                 
        