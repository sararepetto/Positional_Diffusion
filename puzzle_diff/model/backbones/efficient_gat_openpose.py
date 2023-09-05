import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm
from .Transformer_GNN import Transformer_GNN
import sys 
import os 


sys.path.append('puzzle_diff/model/backbones/OpenPose/openpose')
from OpenPose_model import embedding


class Eff_GAT_POSE(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(self, steps, input_channels=2, output_channels=2) -> None:
        super().__init__()
        self.visual_backbone = embedding ()
        self.input_channels = input_channels
        self.output_channels = output_channels
        # visual_feats = 448  # hardcoded

        self.combined_features_dim = 256 + 32 + 32
        #self.output_feature_dimension = 11376
        
         #visual features + pos_feats + temp_feats

        # self.gnn_backbone = torch_geometric.nn.models.GAT(
        #     in_channels=self.combined_features_dim,
        #     hidden_channels=256,
        #     num_layers=2,
        #     out_channels=self.combined_features_dim,
        # )
        self.conv = nn.Conv2d(19,12,1)

        self.gnn_backbone = Transformer_GNN(
            #self.output_feature_dimension,
            self.combined_features_dim,
            hidden_dim=32 * 8,
            heads=8,
            output_size = self.combined_features_dim,
            #output_size=self.output_feature_dimension,
        )
        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        # self.GN = GraphNorm(self.combined_features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
            #nn.Linear(128, self.output_feature_dimension),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim,32),
            #nn.Linear(self.output_feature_dimension, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )
        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch):
        # patch_rgb = (patch_rgb - self.mean) / self.std

        ## fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        # patch_feats = patch_feats

        patch_feats = self.visual_features(patch_rgb)
        final_feats = self.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, patch_feats=patch_feats, batch=batch
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch,
    ):
        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32
       
        # COMBINE  and transform with MLP
        #breakpoint()
        patch_feats = patch_feats.to('cuda:0')
    
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        
        combined_feats = self.mlp(combined_feats)
        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats
        )  # combined -> (err_x, err_y)

        return final_feats

    def visual_features(self, patch_rgb):
        feats = self.visual_backbone.forward(patch_rgb)
        feats = feats.to('cuda:0')
        #feats = self.conv(feats)
        feats = torch.flatten(feats, start_dim =1)
        
        return feats
