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

class Video_dataset(pyg_data.Dataset):
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
        frames,action = self.dataset_get_fn(self.dataset[idx]) # Fx C x W x H
        #frames = torch.stack(frames)
        frames = torch.cat([self.transforms(img)[None, :] for img in frames.numpy()])
        #frames = torch.cat([img[None, :] for img in frames])#-> quando voglio lanciare 
        x = torch.linspace(-1, 1, len(frames))

        adj_mat = torch.ones(len(frames), len(frames))
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=x[:, None],
            frames=frames,
            edge_index=edge_index,
            action = action, 
            ind_name=torch.tensor([idx]).long(),
            num_frames=torch.tensor([len(frames)]),
        )
        return data
    


if __name__ == "__main__":
    from PennAction_RGB_dt import PennAction_RGB_dt

    train_dt = PennAction_RGB_dt(train = True)
    dt = Video_dataset(train_dt, dataset_get_fn=lambda x: x)

    dl = torch_geometric.loader.DataLoader(dt, batch_size=10)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
        print(k)

        