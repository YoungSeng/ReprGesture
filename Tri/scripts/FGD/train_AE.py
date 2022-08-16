import glob
import os
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from embedding_net import EmbeddingNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import sys

[sys.path.append(i) for i in ['.', '..']]
from data_loader.lmdb_data_loader import *


from config.parse_args import parse_args
args = parse_args()

from scipy.spatial.transform import Rotation as R

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optim):
    # zero gradients
    optim.zero_grad()

    # reconstruction loss
    feat, recon_data = net(target_data)
    # pdb.set_trace()
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict


# def make_tensor(path, n_frames, stride=None):
#     mean_vec = np.array([0.00000, 2.52148, -2.91992, 0.99912, -0.01282, -0.00026, 0.01280, 0.99841, -0.03184, 0.00069, 0.03168, 0.99929, 0.00000, 10.86047, 2.18604, 0.99891, -0.01064, 0.01231, 0.01214, 0.98814, -0.12365, -0.01075, 0.12360, 0.98887, 0.00000, 10.34519, -4.27300, 0.99556, -0.01292, 0.03683, 0.01154, 0.99658, 0.03790, -0.03729, -0.03911, 0.99691, 0.00000, 18.38247, -1.77174, 0.99897, -0.01094, 0.01075, 0.01092, 0.99911, 0.00198, -0.01078, -0.00188, 0.99992, 2.80078, 7.18359, 9.08594, 0.99081, -0.00965, 0.00690, 0.00809, 0.99259, 0.03236, -0.01181, -0.03151, 0.99209, 0.38099, -4.21559, -0.46325, 0.19544, 0.59988, -0.76141, -0.94797, 0.23593, -0.06408, 0.14209, 0.74542, 0.63287, 0.00000, 0.00000, 0.00000, 0.99973, 0.00481, -0.01504, -0.01559, 0.76747, -0.37883, 0.00254, 0.37912, 0.76751, 25.57812, -0.59326, -0.68848, 0.66899, -0.42074, -0.25914, 0.41705, 0.76207, -0.15404, 0.26504, -0.14365, 0.90685, 24.78125, -0.57471, -0.66699, 0.99750, -0.01625, 0.02095, -0.02628, 0.80271, 0.13050, -0.00780, -0.13193, 0.80063, 2.06055, 0.07483, 1.81543, 0.93129, 0.05312, -0.02720, -0.05045, 0.94833, 0.03660, 0.02873, -0.00391, 0.97994, -2.80078, 7.18359, 9.08594, 0.99174, -0.00197, -0.02820, 0.00672, 0.99154, 0.04914, 0.03461, -0.04646, 0.99165, -0.46053, -4.15204, -0.50268, 0.14787, -0.28553, 0.93628, 0.95633, 0.19541, -0.09572, -0.15723, 0.92578, 0.31184, 0.00000, 0.00000, 0.00000, 0.99983, 0.00138, 0.00615, 0.00623, 0.86439, -0.09220, 0.00097, 0.09240, 0.86438, -25.57812, -0.71777, -0.67236, 0.70260, 0.39597, 0.07393, -0.39359, 0.71414, -0.06586, -0.08571, -0.04958, 0.98822, -24.78125, -0.69531, -0.65137, 0.99653, 0.01511, -0.05767, 0.04401, 0.67902, 0.47884, 0.02813, -0.48158, 0.67704, -1.98047, -0.26611, 1.88672, 0.93157, -0.10677, 0.06271, 0.11544, 0.92725, -0.17312, -0.03139, 0.18842, 0.96776, 0.00000, 12.78896, 3.29967, 0.98774, 0.00215, 0.00108, -0.00157, 0.97600, -0.00683, -0.00190, -0.00306, 0.98297, 0.00000, 6.35792, 3.19266, 0.98250, 0.01986, -0.09621, -0.02072, 0.98458, -0.05006, 0.09478, 0.05704, 0.96878])
#     std_vec = np.array([0.00000, 0.00000, 0.00000, 0.00099, 0.04008, 0.00314, 0.03994, 0.00134, 0.02021, 0.00451, 0.02018, 0.00092, 0.00000, 0.14517, 0.00929, 0.00117, 0.04129, 0.01479, 0.04060, 0.01405, 0.07932, 0.01665, 0.07927, 0.01408, 0.00000, 0.14244, 0.02803, 0.00754, 0.06749, 0.05224, 0.06825, 0.00244, 0.02440, 0.05124, 0.02391, 0.00720, 0.00000, 0.15541, 0.03031, 0.00108, 0.04099, 0.01242, 0.04098, 0.00099, 0.00146, 0.01242, 0.00135, 0.00049, 0.00000, 0.00000, 0.00000, 0.01770, 0.09248, 0.09636, 0.08873, 0.02407, 0.07221, 0.09951, 0.06753, 0.01339, 0.56550, 0.31915, 0.29897, 0.13823, 0.04766, 0.02888, 0.05802, 0.15796, 0.11513, 0.14312, 0.04008, 0.03919, 0.00000, 0.00000, 0.00000, 0.00041, 0.00693, 0.01700, 0.01746, 0.30126, 0.41975, 0.00566, 0.42006, 0.30120, 0.00000, 0.00000, 0.00000, 0.47124, 0.25131, 0.15194, 0.24654, 0.33872, 0.21469, 0.15956, 0.20909, 0.13262, 0.00000, 0.00000, 0.00000, 0.00256, 0.02132, 0.06213, 0.03357, 0.19991, 0.54484, 0.05602, 0.54550, 0.20201, 0.00000, 0.00000, 0.00000, 0.13813, 0.28839, 0.16395, 0.27264, 0.11446, 0.09677, 0.18949, 0.03967, 0.03751, 0.00000, 0.00000, 0.00000, 0.02189, 0.09414, 0.07947, 0.08226, 0.03106, 0.08155, 0.08926, 0.06965, 0.02171, 0.67348, 0.27224, 0.01710, 0.13154, 0.04565, 0.02449, 0.07059, 0.13398, 0.12308, 0.12868, 0.05645, 0.03498, 0.00000, 0.00000, 0.00000, 0.00037, 0.00933, 0.01654, 0.01615, 0.23627, 0.43385, 0.00999, 0.43405, 0.23629, 0.00000, 0.00000, 0.00000, 0.49426, 0.31135, 0.05373, 0.30777, 0.47508, 0.10175, 0.07146, 0.09017, 0.01958, 0.00000, 0.00000, 0.00000, 0.00419, 0.01920, 0.05474, 0.05844, 0.37313, 0.40627, 0.02783, 0.40886, 0.37545, 0.00000, 0.00000, 0.00000, 0.10151, 0.28821, 0.15324, 0.28953, 0.09001, 0.07058, 0.15410, 0.04622, 0.03273, 0.00000, 0.12843, 0.02536, 0.01814, 0.13863, 0.06960, 0.13529, 0.02434, 0.16875, 0.07589, 0.16612, 0.02035, 0.00000, 0.11726, 0.05886, 0.02845, 0.03148, 0.15255, 0.04438, 0.01287, 0.15983, 0.15010, 0.16066, 0.02975])
#
#     if os.path.isdir(path):
#         files = glob.glob(os.path.join(path, '*.npz'))
#     else:
#         files = [path]
#
#     samples = []
#     stride = n_frames // 2 if stride is None else stride
#     for file in files:
#         data = np.load(file)
#         data = data['clips']
#         for i in range(0, len(data) - n_frames, stride):
#             sample = data[i:i+n_frames]
#             sample = (sample - mean_vec) / std_vec
#             samples.append(sample)
#
#     return torch.Tensor(samples)


def make_tensor(path, n_frames, stride=None):
    # mean_vec = np.array([0.00000, 2.52148, -2.91992, 0.99912, -0.01282, -0.00026, 0.01280, 0.99841, -0.03184, 0.00069, 0.03168, 0.99929, 0.00000, 10.86047, 2.18604, 0.99891, -0.01064, 0.01231, 0.01214, 0.98814, -0.12365, -0.01075, 0.12360, 0.98887, 0.00000, 10.34519, -4.27300, 0.99556, -0.01292, 0.03683, 0.01154, 0.99658, 0.03790, -0.03729, -0.03911, 0.99691, 0.00000, 18.38247, -1.77174, 0.99897, -0.01094, 0.01075, 0.01092, 0.99911, 0.00198, -0.01078, -0.00188, 0.99992, 2.80078, 7.18359, 9.08594, 0.99081, -0.00965, 0.00690, 0.00809, 0.99259, 0.03236, -0.01181, -0.03151, 0.99209, 0.38099, -4.21559, -0.46325, 0.19544, 0.59988, -0.76141, -0.94797, 0.23593, -0.06408, 0.14209, 0.74542, 0.63287, 0.00000, 0.00000, 0.00000, 0.99973, 0.00481, -0.01504, -0.01559, 0.76747, -0.37883, 0.00254, 0.37912, 0.76751, 25.57812, -0.59326, -0.68848, 0.66899, -0.42074, -0.25914, 0.41705, 0.76207, -0.15404, 0.26504, -0.14365, 0.90685, 24.78125, -0.57471, -0.66699, 0.99750, -0.01625, 0.02095, -0.02628, 0.80271, 0.13050, -0.00780, -0.13193, 0.80063, 2.06055, 0.07483, 1.81543, 0.93129, 0.05312, -0.02720, -0.05045, 0.94833, 0.03660, 0.02873, -0.00391, 0.97994, -2.80078, 7.18359, 9.08594, 0.99174, -0.00197, -0.02820, 0.00672, 0.99154, 0.04914, 0.03461, -0.04646, 0.99165, -0.46053, -4.15204, -0.50268, 0.14787, -0.28553, 0.93628, 0.95633, 0.19541, -0.09572, -0.15723, 0.92578, 0.31184, 0.00000, 0.00000, 0.00000, 0.99983, 0.00138, 0.00615, 0.00623, 0.86439, -0.09220, 0.00097, 0.09240, 0.86438, -25.57812, -0.71777, -0.67236, 0.70260, 0.39597, 0.07393, -0.39359, 0.71414, -0.06586, -0.08571, -0.04958, 0.98822, -24.78125, -0.69531, -0.65137, 0.99653, 0.01511, -0.05767, 0.04401, 0.67902, 0.47884, 0.02813, -0.48158, 0.67704, -1.98047, -0.26611, 1.88672, 0.93157, -0.10677, 0.06271, 0.11544, 0.92725, -0.17312, -0.03139, 0.18842, 0.96776, 0.00000, 12.78896, 3.29967, 0.98774, 0.00215, 0.00108, -0.00157, 0.97600, -0.00683, -0.00190, -0.00306, 0.98297, 0.00000, 6.35792, 3.19266, 0.98250, 0.01986, -0.09621, -0.02072, 0.98458, -0.05006, 0.09478, 0.05704, 0.96878])
    # std_vec = np.array([0.00000, 0.00000, 0.00000, 0.00099, 0.04008, 0.00314, 0.03994, 0.00134, 0.02021, 0.00451, 0.02018, 0.00092, 0.00000, 0.14517, 0.00929, 0.00117, 0.04129, 0.01479, 0.04060, 0.01405, 0.07932, 0.01665, 0.07927, 0.01408, 0.00000, 0.14244, 0.02803, 0.00754, 0.06749, 0.05224, 0.06825, 0.00244, 0.02440, 0.05124, 0.02391, 0.00720, 0.00000, 0.15541, 0.03031, 0.00108, 0.04099, 0.01242, 0.04098, 0.00099, 0.00146, 0.01242, 0.00135, 0.00049, 0.00000, 0.00000, 0.00000, 0.01770, 0.09248, 0.09636, 0.08873, 0.02407, 0.07221, 0.09951, 0.06753, 0.01339, 0.56550, 0.31915, 0.29897, 0.13823, 0.04766, 0.02888, 0.05802, 0.15796, 0.11513, 0.14312, 0.04008, 0.03919, 0.00000, 0.00000, 0.00000, 0.00041, 0.00693, 0.01700, 0.01746, 0.30126, 0.41975, 0.00566, 0.42006, 0.30120, 0.00000, 0.00000, 0.00000, 0.47124, 0.25131, 0.15194, 0.24654, 0.33872, 0.21469, 0.15956, 0.20909, 0.13262, 0.00000, 0.00000, 0.00000, 0.00256, 0.02132, 0.06213, 0.03357, 0.19991, 0.54484, 0.05602, 0.54550, 0.20201, 0.00000, 0.00000, 0.00000, 0.13813, 0.28839, 0.16395, 0.27264, 0.11446, 0.09677, 0.18949, 0.03967, 0.03751, 0.00000, 0.00000, 0.00000, 0.02189, 0.09414, 0.07947, 0.08226, 0.03106, 0.08155, 0.08926, 0.06965, 0.02171, 0.67348, 0.27224, 0.01710, 0.13154, 0.04565, 0.02449, 0.07059, 0.13398, 0.12308, 0.12868, 0.05645, 0.03498, 0.00000, 0.00000, 0.00000, 0.00037, 0.00933, 0.01654, 0.01615, 0.23627, 0.43385, 0.00999, 0.43405, 0.23629, 0.00000, 0.00000, 0.00000, 0.49426, 0.31135, 0.05373, 0.30777, 0.47508, 0.10175, 0.07146, 0.09017, 0.01958, 0.00000, 0.00000, 0.00000, 0.00419, 0.01920, 0.05474, 0.05844, 0.37313, 0.40627, 0.02783, 0.40886, 0.37545, 0.00000, 0.00000, 0.00000, 0.10151, 0.28821, 0.15324, 0.28953, 0.09001, 0.07058, 0.15410, 0.04622, 0.03273, 0.00000, 0.12843, 0.02536, 0.01814, 0.13863, 0.06960, 0.13529, 0.02434, 0.16875, 0.07589, 0.16612, 0.02035, 0.00000, 0.11726, 0.05886, 0.02845, 0.03148, 0.15255, 0.04438, 0.01287, 0.15983, 0.15010, 0.16066, 0.02975])

    files = glob.glob(os.path.join(path, '*.npy'))

    samples = []
    stride = n_frames // 2 if stride is None else stride
    for file in files:
        out_data = np.expand_dims(np.load(file), axis=0)
        print(out_data.shape)       # (1, 1889, 108)
        # euler -> rotation matrix
        out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
        out_matrix = np.zeros(
            (out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
        for i in range(out_data.shape[0]):  # mirror
            for j in range(out_data.shape[1]):  # frames
                for k in range(out_data.shape[2]):  # joints
                    out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                    r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                    out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
        out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))
        data = out_matrix[0]
        print(data.shape)
        for i in range(0, len(data) - n_frames, stride):
            sample = data[i:i+n_frames]
            # sample = (sample - mean_vec) / std_vec
            samples.append(sample)

    return torch.Tensor(samples)


def main(n_frames):
    # dataset
    """
    python FGD/train_AE.py --config=<..your path/GENEA/genea_challenge_2022/baselines/Tri/config/multimodal_context.yml>
    """
    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    import datetime
    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path('<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/FGD>').parent / 'tensorboard_runs' / tb_path))

    from utils.vocab_utils import build_vocab
    train_dataset = TwhDataset(args.train_data_path[0],
                               n_poses=args.n_poses,
                               subdivision_stride=args.subdivision_stride,
                               pose_resampling_fps=args.motion_resampling_framerate,
                               data_mean=args.data_mean, data_std=args.data_std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=word_seq_collate_fn
                              )

    val_dataset = TwhDataset(args.val_data_path[0],
                             n_poses=args.n_poses,
                             subdivision_stride=args.subdivision_stride,
                             pose_resampling_fps=args.motion_resampling_framerate,
                             data_mean=args.data_mean, data_std=args.data_std)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=word_seq_collate_fn
                             )

    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train_dataset = TensorDataset(make_tensor('data/Trinity', n_frames))
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)

    # init model and optimizer
    pose_dim = 18 * 12
    generator = EmbeddingNet(pose_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # training
    global_iter = 0
    for epoch in range(100):
        for iter_idx, target in enumerate(train_loader, 0):
            global_iter += 1
            # word_seq, words_lengths, extended_word_seq, poses_seq, audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data
            target = target[3]
            batch_size = target.size(0)
            target_vec = target.to(device)
            loss = train_iter(target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)
                    # write to tensorboard
                    tb_writer.add_scalar(name + '/train', loss[name], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

    # save model
    gen_state_dict = generator.state_dict()
    save_name = f'<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/FGD/model_checkpoint_{n_frames}_100.bin>'
    torch.save({'pose_dim': pose_dim, 'n_frames': n_frames, 'gen_dict': gen_state_dict}, save_name)


if __name__ == '__main__':
    n_frames = 100  # 30, 60, 90
    main(n_frames)
