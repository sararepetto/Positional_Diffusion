import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm

from .Transformer_GNN import Transformer_GNN
from .C3D import C3D

class Eff_GAT(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(self, steps, input_channels=2, output_channels=2, finetuning=False,phase=False) -> None:
        super().__init__()
        self.finetuning = finetuning
        self.phase = phase
        #self.visual_backbone = timm.create_model(
            #"efficientnet_b0", pretrained= True, features_only=True
        #)
        self.visual_backbone = C3D(phase=phase)

        self.input_channels = input_channels
        self.output_channels = output_channels
        # visual_feats = 448  # hardcoded

        self.combined_features_dim = 512 + 32 + 32 #visual features + pos_feats + temp_feats

        # self.gnn_backbone = torch_geometric.nn.models.GAT(
        #     in_channels=self.combined_features_dim,
        #     hidden_channels=256,
        #     num_layers=2,
        #     out_channels=self.combined_features_dim,
        # )
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
        # self.GN = GraphNorm(self.combined_features_dim)
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
        self.pool = nn.AdaptiveAvgPool3d(1)

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
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)
        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        #una feature per ogni frames
        #mean per video

        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats
        )  # combined -> (err_x, err_y)
        #posizione
        return final_feats
        
    def forward_with_embedding(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
    ): 
        #new_t = torch.gather(t, 0, batch.batch)
        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32
        # COMBINE  and transform with MLP
        patch_feats = self.visual_features(patch_feats)
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)
        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        return feats

    def visual_features(self, patch_rgb):
        #patch_rgb = patch_rgb.permute(0,3,1,2)
        patch_rgb = (patch_rgb - self.mean) / self.std
        patch_rgb = patch_rgb.permute(0,2,1,3,4)
        if self.finetuning == False:
            with torch.no_grad():         
                patch_feats = self.visual_backbone.forward(patch_rgb)
                #patch_feats = torch.cat(
                    #[
                       # feats[1].reshape(patch_rgb.shape[0], -1),
                       # feats[2].reshape(patch_rgb.shape[0], -1),
                    #],
                    #-1,
                #)
        else:
            patch_feats = self.visual_backbone.forward(patch_rgb)
            #patch_feats = torch.cat(
                #[
                   # feats[2].reshape(patch_rgb.shape[0], -1),
                    #feats[3].reshape(patch_rgb.shape[0], -1),
                #],
                #-1,
            #)
                #[
                   #feats[1].reshape(patch_rgb.shape[0], -1),
                   # feats[2].reshape(patch_rgb.shape[0], -1),
                #],
                #-1,
            #)
        
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        if self.phase==True:
            patch_feats = patch_feats.permute(0,2,1,3,4)
        #rifare permute per aver clip,channels,frames.w,h
            patch_feats = self.pool(patch_feats)
            patch_feats = patch_feats.view(-1,512)
        return patch_feats
