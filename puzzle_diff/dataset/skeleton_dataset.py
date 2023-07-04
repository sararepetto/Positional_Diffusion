import argparse
import math
import random
from typing import List, Tuple

# import albumentations
# import cv2
import einops
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


class Skeleton_dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=lambda x: x,
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
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
        embedding = self.dataset_get_fn(self.dataset[idx])
        nframes=embedding[0].shape[-1]
        #frames = torch.cat([self.transforms(img)[None, :] for img in images])
        x = torch.linspace(-1, 1,nframes)

        adj_mat = torch.ones(nframes, nframes)
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        data = pyg_data.Data(
            x=x[:, None],
            original=embedding[0].permute(2,0,1),
            perturbed=embedding[2].permute(2,0,1),
            labels=embedding[1],
            edge_index=edge_index,
            #img_path=img_path,
            ind_name=torch.tensor([idx]).long(),
            num_frames=torch.tensor([nframes]),
        )
        return data
    


if __name__ == "__main__":
    
    from NTU_60_dt import NTU_60_dt
    from skeletics_dt import Skeletics_dt
    train_dt = Skeletics_dt()
    dt = Skeleton_dataset(train_dt, dataset_get_fn=lambda x: x)
    dl = torch_geometric.loader.DataLoader(dt, batch_size=2)
    dl_iter = iter(dl)
    k = next(dl_iter)

   
    for i in range(5):
        k = next(dl_iter)
        breakpoint()
        print(k)
        print(k.perturbed)
        print(k.batch)
        print(k.batch.unique)
    pass
