import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import supervision as sv
from torch.utils.data import Dataset
import numpy as np
import torch
import math 
from pathlib import Path
import os 
import glob
import re

class Boxes:
    """Detect objects with yolo mothod (V8)
    """

    def __init__(self):
        """Initialize Detector
        """
        torch.cuda.set_device('cuda:0')
        self.model = YOLO('yolov8x.pt')
        
    def detect(self, image):
        """Detect objects from an image

        Args:
            image (image): input image
        """

        predictions = self.model(image, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(predictions)

        boxes = []
        if len(detections):
            
            for *xyxy, conf, class_id, _ in detections:
                box = xyxy[0]
                if self.model.names[class_id] == 'person':
                    boxes.append([int(box[0]),int(box[1]),int(box[2]),int(box[3])])
         
        boxes = torch.tensor(boxes)
        return boxes
    
    def get_boxes(self):
        self.boxes = Boxes()
        self.data_path = Path("/media/sara/Crucial X6/Ikea/Kallax")
        self.list_files=os.listdir(self.data_path)
        videos_coordinates = []
        zeros = 0
        dev = ['dev1','dev2','dev3']
        coordinate_path = os.listdir('datasets/Ikea')
        ready_files = []
        for i in range (len(coordinate_path)):
            result = coordinate_path[i][15:int(len(coordinate_path[i])-7)]
            ready_files.append(result)
        for i in range(len(self.list_files)):
          if self.list_files[i] != '.~lock.Accordo_affiliatura_09.2022_ITA_REPETTO_UNIPADOVA.docx#':
            for j in range(len(dev)):
                    video_coordinate=[]
                    video_path = f"/media/sara/Crucial X6/Ikea/Kallax/{self.list_files[i]}/{dev[j]}/images/*"                    
                    #video_path = f"/media/sara/Crucial X6/Ikea/Kallax/0008_black_floor_01_01_2019_08_15_11_18/dev1/images/*"
                    video_path = glob.glob(video_path)
                    for t in range(len(video_path)):
                        img = cv2.imread(video_path[t])
                        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.box = self.boxes.detect(image)
                        if self.box.shape == torch.Size([0]):
                            zeros += 1
                            xmin = torch.zeros(1)
                            ymin = torch.zeros(1)
                            xmax = torch.zeros(1)
                            ymax = torch.zeros(1)
                            coordinate = torch.stack((xmin,ymin,xmax,ymax))
                            coordinate = coordinate.view(4)
                        else:
                            xmin = torch.min(self.box[:,0]) - min(torch.min(self.box[:,0]),20)
                            ymin = torch.min(self.box[:,1]) - min(torch.min(self.box[:,1]),20)
                            xmax = torch.max(self.box[:,2]) + min(image.shape[1]-torch.max(self.box[:,2]),20)
                            ymax = torch.max(self.box[:,3]) + min(image.shape[0]-torch.max(self.box[:3]),20)
                            coordinate = torch.stack((xmin,ymin,xmax,ymax))
                        image = image[int(ymin):int(ymax),int(xmin):int(xmax)]
                        video_coordinate.append(coordinate)
            
                     
                    print(self.list_files)
                    print (i)
                    print(dev[j])
                    video_coordinate = torch.stack(video_coordinate, dim = 0)  
                    torch.save(video_coordinate, f'datasets/Ikea/new_coordinates{self.list_files[i]}{dev[j]}.pt')
                    videos_coordinates.append(video_coordinate)

            ##53 frames son senza coordinate 
            #indeces.append(i)
                
        #for i in range(len(video)):
            #video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
            #video[i] = cv2.resize(video[i],(128,128))
            #video[i] = torch.from_numpy(video[i])
        #video = torch.stack(video)
        #video= video[::10]
        
        return videos_coordinates, zeros

    

if __name__ == "__main__":
  boxes = Boxes()
  #img = "/media/sara/Crucial X6/Ikea/Kallax/0001_black_table_02_01_2019_08_16_14_00/dev2/images/000000.jpg"
  #image = cv2.imread(image)
  #
  #plt.imshow(image)
  #plt.show()
  #breakpoint()

  #boxes = Boxes.detect(img)
  new_videos,index = boxes.get_boxes()
  print(index)
