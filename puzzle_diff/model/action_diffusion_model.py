import colorsys
import enum
import math
import pickle
import cv2
# from .backbones.Transformer_GNN import Transformer_GNN
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import scipy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.models
import torchmetrics
import torchvision
import torchvision.transforms.functional as trF
from kornia.geometry.transform import Rotate as krot
from PIL import Image
from scipy.stats import kendalltau
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

import wandb

# from .backbones import Dark_TFConv, Eff_GAT
from .backbones.action_MLP import MLP


matplotlib.use("agg")



class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


def interpolate_color1d(color1, color2, fraction):
    # color1 = [float(x) / 255 for x in color1]
    # color2 = [float(x) / 255 for x in color2]
    hsv1 = color1  #  colorsys.rgb_to_hsv(*color1)
    hsv2 = color2  #  colorsys.rgb_to_hsv(*color2)
    h = hsv1[0] + (hsv2[0] - hsv1[0]) * fraction
    s = hsv1[1] + (hsv2[1] - hsv1[1]) * fraction
    v = hsv1[2] + (hsv2[2] - hsv1[2]) * fraction
    return tuple(x for x in (h, s, v))


def interpolate_color(
    pos, col_1=(1, 0, 0), col_2=(1, 1, 0), col_3=(0, 0, 1), col_4=(0, 1, 0)
):
    f1 = float((pos[0] + 1) / 2)
    f2 = float((pos[1] + 1) / 2)
    c1 = interpolate_color1d(col_1, col_2, f1)
    c2 = interpolate_color1d(col_3, col_4, f1)
    return interpolate_color1d(c1, c2, f2)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cosine_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    # Linear --> every diffusion step 0...T same noise quantity
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out[:, None]  # out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))



class MLP_action(pl.LightningModule):
    def __init__(
        self,
        steps=600,
        inference_ratio=1,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        learning_rate=1e-4,
        sampling="DDPM",
        save_and_sample_every=1000,
        bb=None,
        classifier_free_prob=0,
        classifier_free_w=0,
        noise_weight=0.0,
        warmup_steps=1000,
        max_train_steps=10000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_mean_type = model_mean_type
        self.learning_rate = learning_rate
        self.save_and_sample_every = save_and_sample_every
        self.noise_weight = noise_weight
        self.warmup_steps = warmup_steps
        self.max_train_steps = max_train_steps


        ### DIFFUSION STUFF

        if sampling == "DDPM":
            self.inference_ratio = 1
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddpm,
            )
            self.eta = 1
        elif sampling == "DDIM":
            self.inference_ratio = inference_ratio
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddim,
            )
            self.eta = 0

        # define beta schedule
        betas = linear_beta_schedule(timesteps=steps)
        # self.timesteps = torch.arange(0, 700).flip(0)
        self.register_buffer("betas", betas)
        # self.betas = cosine_beta_schedule(timesteps=steps)
        # define alphas
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.steps = steps
       
        ### BACKBONE
        self.model = MLP(steps=steps, input_dim = 4416, num_classes = 15) 
        self.save_hyperparameters()

    def initialize_torchmetrics(self):
        metrics = {}

        metrics["accuracy"] = torchmetrics.MeanMetric()
        #metrics["tau"] = torchmetrics.MeanMetric()
        metrics["pmr"] = torchmetrics.MeanMetric()
        #metrics["overall_nImages"] = torchmetrics.SumMetric()
        self.metrics = nn.ModuleDict(metrics)
    
    def extract(a, t, x_shape=None):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out[:, None]  # out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, xy_pos, time, rgb_frames, edge_index, batch) -> Any:
        return self.model.forward(xy_pos, time, rgb_frames, edge_index, batch)

    def video_features(self, rgb_frames):
        return self.model.visual_features(rgb_frames)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # x_start + noise depending on t
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(
        self,
        x_start,
        t,
        noise = None,
        loss_type="CE",
        rgb_frames=None,
        edge_index=None,
        target = None, 
        batch=None,
    ):  
        
        if noise is None:
            noise = torch.randn_like(x_start)

        if self.steps == 1:
            x_noisy = torch.zeros_like(x_start)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        
        prediction = self.forward(
            x_noisy,
            t,
            rgb_frames,
            edge_index,
            batch=batch,
        ) # predicted action and compare to the real one
        if loss_type == "CE":
            loss = F.cross_entropy(prediction,target)
        else:
            raise NotImplementedError()

        return loss

   
       
        

    @torch.no_grad()
    def p_sample_ddim(
        self, x, t, t_index, cond, edge_index, video_feats, batch
    ):  

        prev_timestep = t - self.inference_ratio

        eta = self.eta
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (prev_timestep >= 0).all():
            alpha_prod_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prod_prev = alpha_prod * 0 + 1

        beta = 1 - alpha_prod
        beta_prev = 1 - alpha_prod_prev

        model_output = self.forward_with_feats(
                x, t, edge_index=edge_index, video_feats= video_feats, batch=batch
            )

        # estimate x_0
        x_0 = {
            ModelMeanType.EPSILON: (x - beta**0.5 * model_output) / alpha_prod**0.5,
            ModelMeanType.START_X: model_output,
        }[self.model_mean_type]
        eps = self._predict_eps_from_xstart(x, t, x_0)

        variance = self._get_variance(
            t, prev_timestep
        )  # (beta_prev / beta) * (1 - alpha_prod / alpha_prod_prev)
        std_eta = eta * variance**0.5

        # estimate "direction to x_t"
        pred_sample_direction = (1 - alpha_prod_prev - std_eta**2) ** (0.5) * eps

        # x_t-1 = a * x_0 + b * eps
        prev_sample = alpha_prod_prev ** (0.5) * x_0 + pred_sample_direction

        if eta > 0:
            noise = torch.randn(model_output.shape, dtype=model_output.dtype).to(
                self.device
            )
            prev_sample = prev_sample + std_eta * noise
        return prev_sample

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device
        b = shape[0]
        # start from pure noise (for each example in the batch)
        x = torch.randn(shape, device=device) * self.noise_weight

        xs = []

        video_feats = self.video_features(cond)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            x = self.p_sample(
                x,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                feats=video_feats,
                batch=batch,
            )
            
            xs.append(x)
        return xs

    @torch.no_grad()
    def p_sample(
        self, x, t, t_index, cond, edge_index, sampling_func, feats, batch
    ):
        return sampling_func(x, t, t_index, cond, edge_index, feats, batch)

    @torch.no_grad()
    def sample(
        self,
        image_size,
        batch_size=16,
        channels=3,
        cond=None,
        edge_index=None,
        batch=None,
    ):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
            batch=batch,
        )

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters())
        return optimizer


    def training_step(self, batch, batch_idx):
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
        #t=50
        #t=0
        #fare un po' di prove
        new_t = torch.gather(t, 0, batch.batch)

        loss = self.p_losses(
            batch.x,
            new_t,
            loss_type="CE",
            rgb_frames=batch.frames,
            edge_index=batch.edge_index,
            target=batch.action,
            batch=batch.batch,
        )

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, noise = None):
        with torch.no_grad():
            batch_size = batch.batch.max().item() + 1
            t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
            new_t = torch.gather(t, 0, batch.batch)

            if noise is None:
                noise = torch.randn_like(batch.x)

            if self.steps == 1:
                x_noisy = torch.zeros_like(batch.x)
            else:
                x_noisy = self.q_sample(x_start=batch.x, t=new_t, noise=noise)

            batch_size = batch.batch.max().item() + 1
            t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
        
            action = self.forward(
            x_noisy ,
            new_t,
            edge_index=batch.edge_index,
            rgb_frames=batch.frames,
            batch=batch.batch,
        ) 
            n = 0
            for i in range(batch.batch.max() + 1):
                pred=torch.nn.functional.softmax(action, dim= -1)
              
                pred=torch.argmax(pred,dim=-1)
                pred = pred[i]
                device = pred.device
                # Saving entire diff process
                n += 1
                gt = batch.action[i].to(device)
        
                match = (pred == gt).to(
                    batch.action.device
                )
                pmr = match.all().float()
                acc = match.float().mean()
            
                self.metrics["accuracy"].update(acc)
                self.metrics["pmr"].update(pmr)

            self.log_dict(self.metrics)
        

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def save_video(self,
        frames_rgb,
        pos,
        gt_pos,
        file_name: Path,
        ):
        new_frames=frames_rgb[torch.argsort(pos.squeeze()).cpu().numpy()].detach().cpu().numpy()
        new_frames = cv2.normalize(new_frames, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        videos = wandb.Video(new_frames, fps = 1)
        self.logger.experiment.log(
            {f"{file_name.stem}/{self.current_epoch}": videos, "global_step": self.global_step}
        )
       # plt.savefig(f"{file_name}/asd_{self.current_epoch}-{ind_name}.png")
       # plt.close()
        
        writer = cv2.VideoWriter(f"{file_name}/asd_{self.current_epoch}.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30,(128,128))
        for frame in range(len(new_frames)):
            writer.write(frame)
        writer.release()
def kendall_tau(order, ground_truth):
    """
    Computes the kendall's tau metric
    between the predicted sentence order and true order
    Input:
            order: list of ints denoting the predicted output order
            ground_truth: list of ints denoting the true sentence order

    Returns:
            kendall's tau - float
    """

    if len(ground_truth) == 1:
        if ground_truth[0] == order[0]:
            return 1.0
    #chiedere a Gianluca perchè non si può usare il ground_truth direttamente
    #corr, _ = kendalltau(order, ground_truth))
    reorder_dict = {}

    for i in range(len(ground_truth)):
        reorder_dict[ground_truth[i].item()] = i

    new_order = [0] * len(order)
    for i in range(len(new_order)):
        if order[i] in reorder_dict.keys():
            new_order[i] = reorder_dict[order[i]]

    corr, _ = kendalltau(new_order, list(range(len(order))))
    return corr
