import argparse
import math
import os
import random
import warnings
from math import sin, cos, radians
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose



def randominit():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(args.seed)

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        if not torch.is_tensor(val):
            val = torch.tensor(val)
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n



class ToTensor(object):
    def __call__(self, data_numpy):
        return torch.from_numpy(data_numpy)


class Subtract(object):
    def __init__(self, joint=None):
        if joint is None:
            self.joint = random.randint(0, 24)
        else:
            self.joint = joint

    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((C, T, V, M))
        for i in range(V):
            x_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, self.joint, :]
        return x_new


class Zero_out_axis(object):
    def __init__(self, axis=None):
        self.first_axis = axis

    def __call__(self, data_numpy):
        if self.first_axis is not None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0, 2)
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp


class Zero_out_joints(object):
    def __init__(self, joint_list=None, time_range=None):
        self.first_joint_list = joint_list
        self.first_time_range = time_range

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        if self.first_joint_list is not None:
            if isinstance(self.first_joint_list, int):
                all_joints = [i for i in range(V)]
                joint_list_ = random.sample(all_joints, self.first_joint_list)
                joint_list_ = sorted(joint_list_)
            else:
                joint_list_ = self.first_joint_list
        else:
            joints_percentage = .25
            joints_amount = random.randint(1, int(25 * joints_percentage))
            joint_list_ = sorted(random.sample([i for i in range(25)], joints_amount))
        if self.first_time_range is not None:
            if isinstance(self.first_time_range, int):
                all_frames = [i for i in range(T)]
                time_range_ = random.sample(all_frames, self.first_time_range)
                time_range_ = sorted(time_range_)
            else:
                time_range_ = self.first_time_range
        else:
            frame_percentage = .25
            frame_amount = random.randint(1, int(100 * frame_percentage))
            time_range_ = sorted(random.sample([i for i in range(100)], frame_amount))
        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp


class Zero_out_limbs(object):
    def __call__(self, data_numpy):
        limb = random.randint(0, 3)
        if limb == 0:
            limb_joints = list(range(4, 8)) + [21, 22]  # 0 = left arm
        if limb == 1:
            limb_joints = list(range(8, 12)) + [23, 24]  # 1 = right arm
        if limb == 2:
            limb_joints = list(range(12, 16))  # 2 = left leg
        if limb == 3:
            limb_joints = list(range(16, 20))  # 3 = right leg
        temp = data_numpy.copy()
        temp[:, :, limb_joints, :] = 0
        return temp


class Outlier(object):
    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        outlier_joint = random.randint(0, 24)
        outlier_value = random.uniform(-1, 1)
        temp[:, :, outlier_joint, :] = temp[:, :, outlier_joint, :] + outlier_value
        return temp


class Gaus_noise(object):
    def __init__(self, mean=0, std=0.05):
        self.mean = 0 if mean is None else mean
        self.std = 0.05 if std is None else std

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        return temp + noise


class Shear(object):
    def __init__(self, s1=None, s2=None):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        if self.s1 is not None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        if self.s2 is not None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        R = np.array([[1, s1_list[0], s2_list[0]], [s1_list[1], 1, s2_list[1]], [s1_list[2], s2_list[2], 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp


class Rotate(object):
    def __init__(self):
        self.axis = None
        if 'x' in args.aug:
            self.axis = 0
        if 'y' in args.aug:
            self.axis = 1
        if 'z' in args.aug:
            self.axis = 2

    def __call__(self, data_numpy):
        if 'rand' in args.aug:
            self.axis = random.randint(0, 2)
        temp = data_numpy.copy()
        if self.axis == 0:
            angle_x = random.randint(-30, 30)
            alpha = radians(angle_x)
            R = np.array([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]])
            args.rot_angle = [angle_x, 0, 0]
        if self.axis == 1:
            angle_y = random.randint(-30, 30)
            beta = radians(angle_y)
            R = np.array([[cos(beta), 0, -sin(beta)], [0, 1, 0], [sin(beta), 0, cos(beta)]])
            args.rot_angle = [0, angle_y, 0]
        if self.axis == 2:
            angle_z = random.randint(0, 360)
            gamma = radians(angle_z)
            R = np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]])
            args.rot_angle = [0, 0, angle_z]
        if 'all' in args.aug:
            angle_x = random.randint(-30, 30)
            alpha = radians(angle_x)
            angle_y = random.randint(-30, 30)
            beta = radians(angle_y)
            angle_z = random.randint(0, 360)
            gamma = radians(angle_z)
            sin_alpha = sin(alpha)
            cos_alpha = cos(alpha)
            sin_beta = sin(beta)
            cos_beta = cos(beta)
            sin_gamma = sin(gamma)
            cos_gamma = cos(gamma)
            a1 = (cos_beta * cos_gamma)
            a2 = -(cos_alpha * sin_gamma) + (sin_alpha * sin_beta * cos_gamma)
            a3 = (sin_alpha * sin_gamma) + (cos_alpha * sin_beta * cos_gamma)
            b1 = (cos_beta * sin_gamma)
            b2 = (cos_alpha * cos_gamma) + (sin_alpha * sin_beta * sin_gamma)
            b3 = -(sin_alpha * cos_gamma) + (cos_alpha * sin_beta * sin_gamma)
            c1 = -sin_beta
            c2 = (sin_alpha * cos_beta)
            c3 = (cos_alpha * cos_beta)
            R = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
            args.rot_angle = [angle_x, angle_y, angle_z]
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp


class Reverse(object):
    def __init__(self):
        self.list = list(reversed([i for i in range(args.nframes)]))

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        temp = temp[:, :, self.list]
        return temp


def aug_transform(aug_name):
    aug_name_list = aug_name.split("_")
    transform_aug = []
    if aug_name_list[0] != 'None':
        for i, aug in enumerate(aug_name_list):
            augmentation = aug_look(aug)
            if augmentation is not None:
                transform_aug.append(augmentation)
            else:
                continue
    transform_aug.extend([ToTensor(), ])
    transform_aug = Compose(transform_aug)
    return transform_aug


########################################################################################################################
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.mlp_dropout)
        self.layer = []
        if not len(args.mlp_layers) == 1:
            for i in range(len(args.mlp_layers)):
                io = [args.ls, args.mlp_layers[i]] if i == 0 else [args.mlp_layers[i - 1], args.mlp_layers[i]]
                self.layer.append(nn.Linear(io[0], io[1]))
        self.layer.append(nn.Linear(args.mlp_layers[-1], args.num_classes))
        self.layer = nn.ModuleList(self.layer)

    def forward(self, x):
        if not len(args.mlp_layers) == 1:
            for i in range(len(args.mlp_layers)):
                x = self.relu(self.dropout(self.layer[i](x)))
        x = self.layer[-1](x)
        return x


########################################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, extra_pos_tokens=0):
        super(PositionalEncoding, self).__init__()
        self.extra_pos_tokens = extra_pos_tokens
        self.dropout = nn.Dropout(p=args.pos_dropout)
        pe = torch.zeros(args.max_len, args.ls)
        position = torch.arange(0, args.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.ls, 2).float() * (-np.log(10000.0) / args.ls))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # [args.max_len, 1, args.ls]
        if extra_pos_tokens != 0:
            self.pe_rot = nn.Parameter(torch.zeros(extra_pos_tokens, args.ls).unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x[self.extra_pos_tokens:] = x[self.extra_pos_tokens:] + self.pe
        if self.extra_pos_tokens != 0:
            x[:self.extra_pos_tokens] = x[:self.extra_pos_tokens] + self.pe_rot
        return self.dropout(x)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_lambdas = {}
        self.test_lambdas = {}
        self._matching_ = {}
        ################################################################################################################
        # Rotation heads
        self.extra_pos_tokens = 3 if args.rot_heads else 0
        if args.rot_heads:
            self.pre_logits_rot = nn.Identity()
            self.rot_head_x = nn.Linear(args.ls, 61)
            self.rot_head_y = nn.Linear(args.ls, 61)
            self.rot_head_z = nn.Linear(args.ls, 361)
            self.rot_token_x = nn.Parameter(torch.zeros(1, 1, args.ls))
            self.rot_token_y = nn.Parameter(torch.zeros(1, 1, args.ls))
            self.rot_token_z = nn.Parameter(torch.zeros(1, 1, args.ls))
            self.train_lambdas['Train_rot'] = 1e-4
            self.test_lambdas['Test_rot'] = 1e-4
            self._matching_['Train_rot'] = self.compute_rot_ce_loss
            self._matching_['Test_rot'] = self.compute_rot_ce_loss
        # Contrastive learning for motion
        if args.contrastive:
            self.triplet_loss = nn.TripletMarginLoss().cuda()
            self.train_lambdas['Train_Contr'] = 1e-2
            self.test_lambdas['Test_Contr'] = 1e-2
            self._matching_['Train_Contr'] = self.compute_triplet_loss
            self._matching_['Test_Contr'] = self.compute_triplet_loss
        ################################################################################################################
        self.train_lambdas['Train_MSE'] = 1.0
        self.test_lambdas['Test_MSE'] = 1.0
        self._matching_['Train_MSE'] = self.compute_mse_loss
        self._matching_['Test_MSE'] = self.compute_mse_loss
        self.train_losses = list(self.train_lambdas) + ['Train_Loss']
        self.test_losses = list(self.test_lambdas) + ['Test_Loss']
        self.ff_size = args.ls * args.heads
        self.activation = 'gelu'
        self.input_feats = args.njoints * args.nfeats
        ################################################################################################################
        # Skeleton embedding
        self.skelEmbedding = nn.Linear(self.input_feats, args.ls)
        ################################################################################################################
        # Positional embedding
        self.sequence_pos_encoder = PositionalEncoding(self.extra_pos_tokens)
        ################################################################################################################
        # Encoder init
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=args.ls, nhead=args.heads,
                                                          dim_feedforward=self.ff_size, dropout=args.pos_dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=args.layers)
        ################################################################################################################
        # Decoder init
        self.sequence_pos_decoder = PositionalEncoding()
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=args.ls, nhead=args.heads,
                                                          dim_feedforward=self.ff_size, dropout=args.pos_dropout,
                                                          activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=args.layers)
        self.finallayer = nn.Linear(args.ls, self.input_feats)

    def encoder(self, batch):
        x = batch['x'] if args.aug == 'None' else batch['aug']  # [S, J, C, T]
        bs, njoints, nfeats, nframes = x.shape
        ################################################################################################################
        # embedding of the skeleton
        x = x.permute((3, 0, 2, 1)).reshape(nframes, bs, njoints * nfeats)  # [T, S, C*J]
        x = self.skelEmbedding(x)  # [T, S, Ls]
        ################################################################################################################
        # add positional encoding
        if args.rot_heads:
            rot_token_x = self.rot_token_x.expand(-1, x.shape[1], -1)  # [1 1 Ls] -> [1 S Ls]
            rot_token_y = self.rot_token_y.expand(-1, x.shape[1], -1)  # [1 1 Ls] -> [1 S Ls]
            rot_token_z = self.rot_token_z.expand(-1, x.shape[1], -1)  # [1 1 Ls] -> [1 S Ls]
            x = torch.cat((rot_token_x, rot_token_y, rot_token_z, x), dim=0)  # stack rot token on top of pos_embed
        x = self.sequence_pos_encoder(x)  # [T, S, Ls]
        ################################################################################################################
        # transformer layers
        final = self.seqTransEncoder(x)  # [T, S, Ls]
        x_rot = None
        if args.rot_heads:
            rot_x = self.pre_logits_rot(final[0, :])  # [S Ls]
            rot_y = self.pre_logits_rot(final[1, :])  # [S Ls]
            rot_z = self.pre_logits_rot(final[2, :])  # [S Ls]
            final = final[self.extra_pos_tokens:, :]  # [T S Ls]
            x_rot = [self.rot_head_x(rot_x), self.rot_head_y(rot_y), self.rot_head_z(rot_z)]  # [S Rots]
        # get the average of the output
        z = final.mean(axis=0)  # [S, Ls]
        return {'z': z, 'rot_hat': x_rot, 'embedding':final}

    def decoder(self, batch):
        z, y = batch['z'], batch['y']  # z [S Ls] -- y [S]
        bs, latent_dim = z.shape
        timequeries = self.sequence_pos_decoder(torch.zeros(args.nframes, bs, latent_dim, device='cuda'))  # T S Ls
        output = self.seqTransDecoder(tgt=timequeries, memory=z[None])  # [T S Ls]
        output = self.finallayer(output).reshape(args.nframes, bs, args.njoints, args.nfeats)  # [T S J C]
        batch['output'] = output.permute(1, 2, 3, 0)  # [S J C T]
        return batch

    def compute_loss(self, batch, step='train'):
        mixed_loss = 0
        losses = {}
        loss_list = self.train_lambdas.items() if step == 'train' else self.test_lambdas.items()
        for ltype, lam in loss_list:
            loss_function = self._matching_[ltype]
            loss = loss_function(batch)
            mixed_loss += loss * lam
            losses[ltype] = loss.item()
        if step == 'train':
            losses['Train_Loss'] = mixed_loss.item()
        else:
            losses['Test_Loss'] = mixed_loss.item()
        return mixed_loss, losses

    @staticmethod
    def compute_mse_loss(batch):
        gt = batch['x'].permute(0, 3, 1, 2)  # [S T J C]
        pred = batch['output'].permute(0, 3, 1, 2)  # [S T J C]
        loss = F.mse_loss(gt, pred, reduction='mean')
        return loss

    @staticmethod
    def compute_rot_ce_loss(batch):
        loss_x = criterion(batch['rot_hat'][0], batch['rot'][:, 0])
        loss_y = criterion(batch['rot_hat'][1], batch['rot'][:, 1])
        loss_z = criterion(batch['rot_hat'][2], batch['rot'][:, 2])
        loss = loss_x + loss_y + loss_z
        return loss

    def compute_triplet_loss(self, batch):
        with torch.no_grad():
            batch_ctr = batch.copy()
            # Store reconstructed joints from vanilla encoder as anchor
            batch_ctr['anchor'] = batch_ctr['z'].clone()
            # Convert reconstructed joints in motion, in forward (positive) and backward (negative) motion
            batch_ctr['motion'] = batch_ctr['x'].clone()
            batch_ctr['reverse'] = batch_ctr['x'].clone().permute(0, 2, 1, 3)[:, :, :, list(reversed(
                [i for i in range(args.nframes)]))]
            batch_ctr['motion'][:, :, :, :-1] = batch_ctr['motion'][:, :, :, 1:] - batch_ctr['motion'][:, :, :, :-1]
            batch_ctr['reverse'][:, :, :, :-1] = batch_ctr['reverse'][:, :, :, 1:] - batch_ctr['reverse'][:, :, :, :-1]
            # Feedforward the model with forward motion data to obtain positive samples
            batch_ctr['x'] = batch_ctr['motion'].clone()
            if args.aug != 'None':
                batch_ctr['aug'] = batch_ctr['motion'].clone()
            batch_ctr.update(self.encoder(batch_ctr))
            batch_ctr['pos'] = batch_ctr['z'].clone()
            # Feedforward the model with forward motion data to obtain negative samples
            batch_ctr['x'] = batch_ctr['reverse'].clone()
            if args.aug != 'None':
                batch_ctr['aug'] = batch_ctr['reverse'].clone()
            batch_ctr.update(self.encoder(batch_ctr))
            batch_ctr['neg'] = batch_ctr['z'].clone()
            # Compute triplet margin loss
            loss = self.triplet_loss(batch_ctr['anchor'], batch_ctr['pos'], batch_ctr['neg'])
        return loss

    def forward(self, batch):
        batch.update(self.encoder(batch))
        batch.update(self.decoder(batch))
        return batch



def parseargs():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--seed', type=int, default=1990, help='random seed (default: 1990)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)')

    # I/O settings
    parser.add_argument('--save_data', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='/media/gpaoletti/2TB/SKELTER/ECCV/saved_files/model')
    parser.add_argument('--data_path', type=str, default='/home/sara/Project/SKELTER/NTU')

    # Model settings
    parser.add_argument('--pos_dropout', type=float, default=0.1, help='position embedding dropout rate (default: 0.1')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads (default: 4)')
    parser.add_argument('--layers', type=int, default=2, help='number of transformer blocks (default: 8)')
    parser.add_argument('--ls', type=int, default=256, help='latent space size (default: 256)')
    parser.add_argument('--max_len', type=int, default=100, help='maximum length of position embedding (default: 5000')
    parser.add_argument('--mlp_dropout', type=float, default=0.1)
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[256, 256, 128])
    parser.add_argument('--mlp_lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')

    # Ablation settings
    parser.add_argument('--split', type=str, default='xview60', choices=['xsub60', 'xview60', 'xsub120', 'xview120'])
    parser.add_argument('--aug', type=str, default='gausNoise',
                        help="choose a single or combos (_-separated) of "
                             " 'gausNoise', 'shear', 'subtract', 'zeroOutAxis', 'zeroOutJoints', 'zeroOutLimbs' "
                             " 'outlier', 'rotate_all', 'rotate_rand', 'contrastive', 'None' ")

    parser.add_argument('--distribution', type=bool, default=True)

    args = parser.parse_args()
    args.nfeats = 3
    args.njoints = 25
    args.nframes = 100

    args.rot_heads = True if 'rotate' in args.aug else False
    args.contrastive = True if args.aug == 'contrastive' else False
    args.aug = 'None' if args.aug == 'contrastive' else args.aug

    return args

from pathlib import Path


if __name__ == '__main__':
    args = parseargs()
    randominit()
    train_loader, test_loader = load_data()
    el= next(iter(train_loader))
    # breakpoint()
    model = DataParallel(CVAE().cuda(), device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss().cuda()
    fname_model = '/home/sara/Project/SKELTER/model/gausNoise.pt'
    model.load_state_dict(torch.load(fname_model), strict=False)
    output=model(el)
    breakpoint()
    
    