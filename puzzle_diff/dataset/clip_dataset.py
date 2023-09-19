import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.loader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import torchvision

class Clip_dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=lambda x: x,
        train=True,
        augmentation= False
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.augmentation = augmentation
        self.dataset_get_fn = dataset_get_fn

        if train==True:
            
            if self.augmentation==True:
                self.transforms = transforms.Compose(
                    [   transforms.ToPILImage(),
                        transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                        transforms.ToTensor(),
            
                    ]
        )   
            else: 
                self.transforms = transforms.Compose(
                [ 
                    transforms.ToTensor(),
            
                ]
            )
                
        else:

            self.transforms = transforms.Compose(
                [   
                    transforms.ToTensor(),
            
                ]
        )


    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        frames,action = self.dataset_get_fn(self.dataset[idx])
        #if train==True:
        #frames = self.dataset_get_fn(self.dataset[idx]) # Fx C x W x H
        #frames = torch.stack(frames)
            #PIL = torchvision.transforms.ToPILImage()
        new_frames = []
        for i in range(len(frames)):
              frame = torch.cat([self.transforms(img)[None, :] for img in frames[i].numpy()])
              new_frames.append(frame)
        new_frames = torch.stack(new_frames)
        #new_frames = new_frames.permute(0,2,1,3,4) 
        #frames = torch.cat([self.transforms(img)[None, :] for img in frames.numpy()])

        #frames = torch.cat([img[None, :] for img in frames])#-> quando voglio lanciare 
        x = torch.linspace(-1, 1, len(new_frames))

        adj_mat = torch.ones(len(new_frames), len(new_frames))
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=x[:, None],
            frames=new_frames,
            edge_index=edge_index,
            action = action, 
            ind_name=torch.tensor([idx]).long(),
            num_frames=torch.tensor([len(frames)]),
        )
        return data
    


if __name__ == "__main__":
    from UCF101_clip_dt import UCF101_clip_dt

    train_dt = UCF101_clip_dt(train = False)
    dt = Clip_dataset(train_dt, dataset_get_fn=lambda x: x, train=True)

    dl = torch_geometric.loader.DataLoader(dt, batch_size=10)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
        print(k)

        