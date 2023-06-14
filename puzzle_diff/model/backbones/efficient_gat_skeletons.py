import math

import timm
import torch
import torch.nn as nn
from torch import Tensor

from .SKELTER import CVAE
from .Transformer_GNN import Transformer_GNN


class Eff_GAT_Skeletons(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, steps, input_channels=1, output_channels=1,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.backbone = CVAE()
        fname_model = './gausNoise.pt'
        #self.backbone.load_state_dict(torch.load(fname_model), strict=False)

        self.combined_features_dim = 256 + 32 + 32
        self.gnn_backbone = Transformer_GNN(
            self.combined_features_dim,
            hidden_dim=32 * 8,
            heads=8,
            output_size=self.combined_features_dim,
        )
        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )


    def forward(self, xy_pos, time, skeletons, edge_index, batch):
        """
        xy_pos: only temporal position
        time: 0...T
        """
        skeleton_feats = self.skeleton_features(skeletons)
        output = self.forward_with_feats(
            xy_pos, time, skeleton_feats, edge_index, batch=batch
        )
        return output

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        skeleton_feats: Tensor,
        edge_index: Tensor,
        batch,
    ):
        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32

        # COMBINE  and transform with MLP

        combined_feats = torch.cat([skeleton_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)

        # Residual + final transform
        output = self.final_mlp(
            feats + combined_feats
        ) # Position of the frame in the video (between -1, +1)

        return output

    def skeleton_features(self, skeletons):
        
        skeleton_features = self.backbone.encoder(skeletons)['embedding']
        batch_size_t = math.prod(skeleton_features.shape[:2])
        skeleton_features = skeleton_features.permute(1, 0, 2).reshape(batch_size_t, -1)
        return skeleton_features
