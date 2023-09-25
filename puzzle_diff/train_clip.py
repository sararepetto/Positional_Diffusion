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
from dataset.dataset_utils import get_dataset_clip
from dataset.dataset_utils import get_dataset_class
from model import spatial_diffusion_clip as sd 
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb
import numpy as np
from sklearn.svm import SVC, LinearSVC
import torch.nn as nn
import torch.optim as optim
import torch 
import torchmetrics

#classe(backbone,num_classes):
#backbone+mlp
class Classification(pl.LightningModule):
    def __init__(self,backbone,num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.linear = nn.Linear(512, self.num_classes)
    
    def initialize_torchmetrics(self):
        metrics = {}

        metrics["accuracy"] = torchmetrics.MeanMetric()
        metrics["tau"] = torchmetrics.MeanMetric()
        metrics["pmr"] = torchmetrics.MeanMetric()
        metrics["overall_nImages"] = torchmetrics.SumMetric()
        self.metrics = nn.ModuleDict(metrics)
        
    def forward(self,x):
        x = x.permute(0,2,1,3,4)
        x = self.backbone.forward(x)
        x = self.linear(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=9e-1, weight_decay=5e-4)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        input = train_batch.frames
        target = train_batch.action
        output = self.forward(input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output,target)
        self.log("action_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            n=0
            input= val_batch.frames
            target= val_batch.action
            output = self.forward(input)
            for i in range(val_batch.batch.max() + 1):
                n+=1
                new_output = output[val_batch.batch == i]
                new_output = torch.mean(new_output, dim = 0)
                pts = torch.argmax(new_output)
                match = (target[i] ==pts)
                acc = match.float().mean()
            #correct += torch.sum(targets == pts).item()
                self.metrics["accuracy"].update(acc)
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
    augmentation
):
    ### Define dataset
    train_dt, val_dt, test_dt = get_dataset_clip(dataset=dataset, augmentation=augmentation)
    train_acc_dt,val_acc_dt, test_acc_dt = get_dataset_class(dataset=dataset, augmentation = augmentation)
    dl_train = torch_geometric.loader.DataLoader(
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    acc_dl_train = torch_geometric.loader.DataLoader(
        train_acc_dt, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    acc_dl_test = torch_geometric.loader.DataLoader(
        test_acc_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
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
        finetuning=finetuning
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
        #accelerator="gpu",
        accelerator = "cpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
        max_epochs = 0
    )
    if evaluate:
        model = sd.GNN_Diffusion.load_from_checkpoint(checkpoint_path)
        model.initialize_torchmetrics()
        model.noise_weight = noise_weight
        trainer.test(model, dl_test)
    else:
        trainer.fit(model, dl_train, dl_test, ckpt_path=checkpoint_path)

    acc_model = Classification(model.model.visual_backbone,num_classes=101)
    acc_model.initialize_torchmetrics()

    trainer_acc = pl.Trainer(
        accelerator="cpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
        max_epochs = 50
    )

    trainer_acc.fit(acc_model,acc_dl_train,acc_dl_test,ckpt_path=checkpoint_path)
    #model.model.visual_backbone->rete addestrata 
    train_data=trainer.predict(model,train_dt)
    train_action=[]
    train_embedding=[]
    for i in range(len(train_data)):
        train_action.append(train_data[i][1])
        train_embedding.append(train_data[i][0])
    train_action = np.concatenate(train_action)
    train_embedding = np.concatenate(train_embedding)

    test_data=trainer.predict(model,test_dt)
    test_action=[]
    test_embedding=[]
    for i in range(len(test_data)):
        test_action.append(test_data[i][1])
        test_embedding.append(test_data[i][0])
    test_action = np.concatenate(test_action)
    test_embedding = np.concatenate(test_embedding)
    

    def fit_svm_model(train_embs, train_labels,
        val_embs, val_labels):
        svm_model = SVC(decision_function_shape='ovo', verbose=0)
        svm_model.fit(train_embs, train_labels)
        train_acc = svm_model.score(train_embs, train_labels)
        val_acc = svm_model.score(val_embs, val_labels)
        return svm_model, train_acc, val_acc
    
    results=fit_svm_model(train_embedding,train_action,test_embedding,test_action)
    
    model.log('svm_test',results[2])
    model.log('svm_train',results[1])

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
        augmentation=args.augmentation
    )
