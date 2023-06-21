import torch
from torch.utils.data import Dataset 

class Skeletics_dt(Dataset):
    def __init__(self,train=True):
        super().__init__()
        if train == False:
            self.data= torch.stack(torch.load('datasets/Skeletics/test_data.pt'))
            self.label=torch.load('datasets/Skeletics/test_label.pt')
        else:
            self.data= torch.stack(torch.load('datasets/Skeletics/train_data.pt'))
            self.label=torch.load('datasets/Skeletics/train_label.pt')

        self.N, _, _, self.T = self.data.shape
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        data = self.data[index].permute(1, 0, 2).float().clone()
        label = self.label[index]
        return data, label, data


if __name__ == '__main__':
    dt = Skeletics_dt()
    x = dt[30]
    breakpoint()