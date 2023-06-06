from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import os



class Assembly_dt(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = Path("/home/sara/Downloads/Online/RGB/")
        self.list_files=os.listdir(self.data_path)
    
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self,idx):
        
        element=self.list_files[idx]
        video_path=f"/home/sara/Downloads/Online/RGB/{element}"
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
            video[i]= video[i][360:,750:1200]
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
            video[i]=cv2.resize(video[i],(64,64))
        return video

if __name__ == "__main__":
        dt = Assembly_dt()
        x = dt[30]
        #frames = torch.cat([self.transforms(img)[None, :] for img in images])
        breakpoint()
        print(len(x))
        plt.figure()
        plt.imshow(x[50])
        plt.show()