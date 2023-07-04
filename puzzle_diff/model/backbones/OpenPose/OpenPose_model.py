import cv2 
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import model
from src import util
from src.body import Body
import torch_geometric.loader
import torch.nn as nn
from PIL import Image

class embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.body_estimation = Body('PT_OpenPose/body_pose_model.pth')
    def forward(self,data):
        self.embedding,candidate,subset= self.body_estimation(data)
        return self.embedding,candidate, subset

if __name__ == '__main__':
    import sys 
    sys.path.append('/home/sara/Project/Positional_Diffusion/puzzle_diff/dataset')
    #img= Image.open("PT_OpenPose/prova.png").convert('RGB')
    #im=np.array(img)
    #breakpoint()
    #body_estimation = Body('PT_OpenPose/body_pose_model.pth')
    #breakpoint()
    from ntu_RGB_dt import ntu_RGB_dt
    from video_dataset import Video_dataset
    train_dt = ntu_RGB_dt()
    dt = Video_dataset(train_dt, dataset_get_fn=lambda x: x)
    dl = torch_geometric.loader.DataLoader(dt, batch_size=2)
    dl_iter = iter(dl)
    k = next(dl_iter)
    img =  np.transpose(k.frames[1].numpy(),(1,2,0))
    #k1= next(dl_iter)
    breakpoint()
    #k1 = next(dl_iter)
    #plt.imshow(np.transpose(k1.frames[1].numpy(),(1,2,0)))
    #cplt.show()
    breakpoint()
    model = embedding()
    #breakpoint()
    #criterion = nn.CrossEntropyLoss().cuda()
    # fname_model = '/home/sara/Project/SKELTER/model/gausNoise.pt'
    # model.load_state_dict(torch.load(fname_model), strict=False)
    #device='cuda'
    output,candidate,subset=model(img)
    #plt.imshow(output[0][1])
    #plt.show()
    #canvas = copy.deepcopy(img)
    breakpoint()
    #canvas = util.draw_bodypose(canvas,candidate,subset)
    #plt.imshow(figsize=(10,10))
    #plt.imshow(canvas[:,:,[2,1,0]])
    #plt.axis('off')
    #plt.show()
    #breakpoint()
