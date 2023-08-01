import torch
import torch.nn as nn
from .efficient_gat_action import New_Eff_GAT

class MLP(nn.Module):
    def __init__(self, steps, input_dim = 4416, num_classes = 15) -> None:
        super().__init__()

        self.backbone = New_Eff_GAT(steps, input_channels=1, output_channels=1)
        self.backbone.load_state_dict(torch.load('NEW_EFF_GAT.pt'))
        self.input_dim = input_dim
        self.output_dim = num_classes 
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.output_dim),
        )

    def get_input(self,xy_pos,time,video_feats,edge_index,batch):
            with torch.no_grad():#?-> per frizzare il diffusion
                return self.backbone.forward( xy_pos, time, video_feats,edge_index, batch)
        
    def forward(self,xy_pos,time,video_feats,edge_index,batch):
            x = self.get_input(xy_pos,time, video_feats,edge_index,batch)
            x = self.mlp(x)
            return x
        


    def visual_features(self,video_feats):
            return self.backbone.visual_features(video_feats)
            

        