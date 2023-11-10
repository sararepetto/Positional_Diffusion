import argparse
import os
import sys

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math
import random
import string

import pytorch_lightning as pl
from dataset.dataset_utils import get_dataset_videos
from model import spatial_diffusion_videos as sd 
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb
import numpy as np
from sklearn.svm import SVC, LinearSVC
import torch
import torch.nn as nn
import torchmetrics
from transformers.optimization import Adafactor
from dataset.diffusionn_embedding import Diffusion_embedding
from torch.utils.data import DataLoader


class Classification(pl.LightningModule):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.linear = nn.Linear(448, self.num_classes)
        #mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        #std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        #self.register_buffer("mean", mean)
        #self.register_buffer("std", std)
        #self.pool = nn.AdaptiveAvgPool3d(1)

    def initialize_torchmetrics(self):
        metrics = {}
        metrics["action_accuracy"] = torchmetrics.MeanMetric()
        #metrics["tau"] = torchmetrics.MeanMetric()
        #metrics["pmr"] = torchmetrics.MeanMetric()
        #metrics["overall_nImages"] = torchmetrics.SumMetric()
        self.metrics = nn.ModuleDict(metrics)
        
    def forward(self,x):
        #x = (x - self.mean) / self.std
        #x = x.permute(0,2,1,3,4)
        #x = self.backbone.forward(x)
        #x = x.permute(0,2,1,3,4)
        #x = self.pool(x)
        #x = x.view(-1,512)
        x = self.linear(x)
        return x
    
    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr=1e-5, momentum=9e-1, weight_decay=5e-4, nesterov = True)
        #optimizer = optim.Adam(self.parameters(), lr= 1e-4)
        optimizer = Adafactor(self.parameters())
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        input = train_batch[0]
        target = train_batch[1]
        output = self.forward(input)
        criterion = nn.CrossEntropyLoss()
        try:
            loss = criterion(output,target)
        except:
            print(target)
            print(output)
        self.log("action_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            n=0
            input= val_batch[0]
            target= val_batch[1]
            idx= val_batch[2]
            output = self.forward(input)
            output_new_dim = (input.shape[0],int(output.shape[0]/input.shape[0]),output.shape[1])
            target_new_dim = (input.shape[0],int(output.shape[0]/input.shape[0]))
            target = target.view(target_new_dim)
            output = output.view(output_new_dim)
            for i in range(int(idx.max().item()) + 1):
                n+=1
                new_output = torch.flatten(output[idx == i],end_dim = 1)
                new_target = torch.flatten(target[idx == i])
                #new_output = torch.mean(new_output, dim = 0)
                pts = torch.argmax(new_output,dim=1)
                match = (new_target ==pts)
                acc = match.float().mean()
            #correct += torch.sum(targets == pts).item()
                self.metrics["action_accuracy"].update(acc)
            self.log_dict(self.metrics)

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics)


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str  # print("Random string of length", length, "is:", result_str)


def main(
    batch_size,
    gpus,
    steps,
    num_workers,
    dataset,
    sampling,
    inference_ratio,
    offline,
    noise_weight,
    checkpoint_path,
    predict_xstart,
    evaluate,
    finetuning,
    subsampling,
    augmentation,
    feature_t,
):
    ### Define dataset
    train_dt, val_dt, test_dt = get_dataset_videos(dataset=dataset,subsampling=subsampling, augmentation=augmentation)

    dl_train = torch_geometric.loader.DataLoader(
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    epoch_steps = len(dl_train) * 10
    max_steps = len(dl_train) * 100

    model = sd.GNN_Diffusion(
        steps=steps,
        sampling=sampling,
        inference_ratio=inference_ratio,
        model_mean_type=sd.ModelMeanType.EPISLON
        if not predict_xstart
        else sd.ModelMeanType.START_X,
        warmup_steps=epoch_steps,
        max_train_steps=max_steps,
        noise_weight=noise_weight,
        finetuning=finetuning,
        feature_t = feature_t,
    )
    model.initialize_torchmetrics()

    ### define training
    experiment_name = f"video-{dataset}-{steps}-{get_random_string(6)}"

    tags = [f"{dataset}", "video", "train"]

    wandb_logger = WandbLogger(
        project="Video-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=offline,
        name=experiment_name,
        entity="puzzle_diff_academic",
        tags=tags,
    )

    checkpoint_callback = ModelCheckpoint(monitor="accuracy", mode="max", save_top_k=2)

    trainer = pl.Trainer(
        #
        accelerator="gpu",
        devices=gpus,
        #accelerator='cpu',
        #devices=cpu,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
        max_epochs = 150,
    )
    if evaluate:
        model = sd.GNN_Diffusion.load_from_checkpoint(checkpoint_path)
        model.initialize_torchmetrics()
        model.noise_weight = noise_weight
        trainer.test(model, dl_test)
    else:
        trainer.fit(model, dl_train, dl_test, ckpt_path=checkpoint_path)
    train_data=trainer.predict(model,dl_train)
    test_data=trainer.predict(model,dl_test)
    train_dt1=Diffusion_embedding(train_data)
    test_dt1=Diffusion_embedding(test_data)

    def my_collate_fn(data):
        x, labels= zip(*data)
        indexes = torch.cat([torch.ones(len(x)) * idx for idx, x in enumerate(labels)])
        labels = torch.transpose(torch.vstack(labels),0,1)[0]
        return torch.vstack(x), labels, indexes

        
    dl_train1 = DataLoader(train_dt1,batch_size=batch_size,collate_fn=my_collate_fn,num_workers=num_workers)
    dl_test1 = DataLoader(test_dt1,batch_size=batch_size,collate_fn=my_collate_fn,num_workers=num_workers)

    acc_checkpoint_callback = ModelCheckpoint(monitor="action_accuracy", mode="max", save_top_k=2)
    acc_model = Classification(num_classes=4)
    acc_model.initialize_torchmetrics()

    trainer_acc = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        #accelerator='cpu',
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=10,
        logger=wandb_logger,
        callbacks=[acc_checkpoint_callback, ModelSummary(max_depth=2)],
        max_epochs = 150,
        gradient_clip_val = 1,
        gradient_clip_algorithm = "value",
        detect_anomaly=True,
    )

    trainer_acc.fit(acc_model,dl_train1,dl_test1,ckpt_path=checkpoint_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=100)
    ap.add_argument("-num_workers", type=int, default=8)
    ap.add_argument("-dataset", default="ntu", choices=["ntu","pennaction","ikea","UCF101"])
    ap.add_argument("-sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("-inference_ratio", type=int, default=10)
    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--predict_xstart", type=bool, default=True)
    ap.add_argument("--evaluate", type=bool, default=False)
    ap.add_argument("--noise_weight", type=float, default=0.0)
    ap.add_argument("--finetuning", type=bool, default=False)
    ap.add_argument("--subsampling", type=int, default = 3)
    ap.add_argument("--augmentation", type=bool, default = False)
    ap.add_argument("--feature_t", type=int, default = 100)
    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
        checkpoint_path=args.checkpoint_path,
        predict_xstart=args.predict_xstart,
        noise_weight=args.noise_weight,
        evaluate=args.evaluate,
        finetuning= args.finetuning,
        subsampling=args.subsampling,
        augmentation=args.augmentation,
        feature_t = args.feature_t
    )
