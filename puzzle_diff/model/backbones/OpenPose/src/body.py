import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import transforms

from src import util
from src.model import bodypose_model

class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale = 0.5
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = scale * boxsize / oriImg.shape[1] 
        heatmap_avg = np.zeros((oriImg.shape[1], oriImg.shape[2], 19))
        paf_avg = np.zeros((oriImg.shape[1], oriImg.shape[2], 38))
        
        
            #scale = multiplier[m]
        new = []
        for i in range(oriImg.shape[0]):
                im = torch.from_numpy(oriImg[i])
                new.append(im)
        data = torch.stack(new).float()
        if torch.cuda.is_available():
                data = data.cuda()
        with torch.no_grad():#da capire con il finetuning come funziona, togliendo torch.no_grad
            Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.detach().cpu().numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.detach().cpu().numpy()
           

        return  Mconv7_stage6_L2
if __name__ == "__main__":
    body_estimation = Body('../model/body_pose_model.pth')

    test_image = '../images/ski.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
