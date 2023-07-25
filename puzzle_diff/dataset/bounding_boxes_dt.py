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
        self.data_path = Path("datasets/NTU_60/RGB_videos/RGB_videos_train")
        self.list_files=os.listdir(self.data_path)
        videos_coordinates = []
        zeros = 0
        for i in self.list_files:
            print(i)
            video_path=f"datasets/NTU_60/RGB_videos/RGB_videos_train/{i}"
            video_coordinate = []
            cap = cv2.VideoCapture(video_path)
            index = 0
            while cap.isOpened():
                Ret, Mat = cap.read()

                if Ret :
                    self.box = self.boxes.detect(Mat)
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
                        xmax = torch.max(self.box[:,2]) + min(Mat.shape[1]-torch.max(self.box[:,2]),20)
                        ymax = torch.max(self.box[:,3]) + min(Mat.shape[0]-torch.max(self.box[:3]),20)
                        coordinate = torch.stack((xmin,ymin,xmax,ymax))
                   
                    video_coordinate.append(coordinate)
                    index += 1
                else:
                    break 
           
            video_coordinate = torch.stack(video_coordinate, dim = 0)  
            torch.save(video_coordinate, f'datasets/NTU_60/new_coordinates{i}.pt')
            videos_coordinates.append(video_coordinate)

            ##53 frames son senza coordinate 
            #indeces.append(i)
            cap.release()
        #for i in range(len(video)):
            #video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
            #video[i] = cv2.resize(video[i],(128,128))
            #video[i] = torch.from_numpy(video[i])
        #video = torch.stack(video)
        #video= video[::10]
        
        return videos_coordinates, index

    

if __name__ == "__main__":
  boxes = Boxes()
  new_videos,index = boxes.get_boxes()
  print(index)
