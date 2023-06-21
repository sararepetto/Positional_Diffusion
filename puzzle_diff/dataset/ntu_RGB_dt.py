from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import os


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
            #video[i]= video[i][360:,750:1200]
           video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
      #      breakpoint()
           video[i]=cv2.resize(video[i],(64,64))
        return video

if __name__ == "__main__":
        dt = ntu_RGB_dt()
        breakpoint()
        frames=0
        #for i in range (len(dt)): 
            #x=dt[i]
            #if len(x) > frames:
                #frames = len(x)
        #breakpoint()
        x= dt[30]
        print(len(x))
        plt.figure()
        plt.imshow(x[50])
        plt.show()