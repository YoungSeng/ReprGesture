import os
import pdb

import numpy as np
import torch

from embedding_space_evaluator import EmbeddingSpaceEvaluator

import sys

[sys.path.append(i) for i in ['.', '..']]
from train_AE import make_tensor

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# from data_loader.lmdb_data_loader import *
#
# from config.parse_args import parse_args
# args = parse_args()


def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    fgd_evaluator.push_real_samples(gt_data)
    fgd_evaluator.push_generated_samples(test_data)
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    return fgd_on_feat, fdg_on_raw


def exp_base(chunk_len):
    # AE model
    ae_path = f'<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/FGD/model_checkpoint_{chunk_len}_100.bin>'
    fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, chunk_len, device)

    # load GT data
    GT_path = "<..your path/GENEA/genea_challenge_2022/dataset/v1_18_1/val/>"
    # gt_data = make_tensor(os.path.join(GT_path, 'npy'), chunk_len).to(device)
    # torch.save(gt_data, GT_path + 'gt_data.pt')
    gt_data = torch.load(GT_path + 'gt_data.pt', map_location=device)

    # load generated data
    print(f'----- Experiment (motion chunk length: {chunk_len}) -----')
    print('FGDs on feature space and raw data space')

    # generated_path = '<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_MISA_48_nodif>'
    # test_data = make_tensor(os.path.join(generated_path, 'npy_FGD'), chunk_len).to(device)
    # torch.save(test_data, generated_path + 'test_data.pt')
    # test_data = torch.load(generated_path + 'test_data.pt')
    test_data = torch.load(GT_path + 'gt_data.pt', map_location=device)

    fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
    print(f'{fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')


def exp_per_seq(chunk_len, stride=1):
    n_test_seq = 10
    systems = ['BA', 'BT', 'SA', 'SB', 'SC', 'SD', 'SE']

    # AE model
    ae_path = f'output/model_checkpoint_{chunk_len}.bin'
    fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, chunk_len, device)

    # run
    print(f'----- Experiment (motion chunk length: {chunk_len}, stride: {stride}) -----')
    print('FGDs on feature space and raw data space for each system and each test speech sequence')
    for system_name in systems:
        results = []
        for i in range(n_test_seq):
            name = f'TestSeq{i+1:03d}'

            # load GT data
            gt_data = make_tensor(os.path.join('data/GroundTruth', name + '.npz'), chunk_len, stride=stride)
            gt_data = gt_data.to(device)
            # print(gt_data.shape)

            # load generated data
            test_data = make_tensor(os.path.join(f'data/Cond_{system_name}', name + '.npz'), chunk_len, stride=stride)
            test_data = test_data.to(device)
            # print(test_data.shape)

            # calculate fgd
            fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
            print(f'Cond_{system_name} {name}: {fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')
            results.append([fgd_on_feat, fgd_on_raw])
        results = np.array(results)
        print(f'Cond_{system_name} M=({np.mean(results[:, 0])}, {np.mean(results[:, 1])}), SD=({np.std(results[:, 0])}, {np.std(results[:, 1])})')


if __name__ == '__main__':
    # calculate fgd per system
    # exp_base(30)
    # exp_base(60)
    # exp_base(90)
    exp_base(100)

    # calculate fgd per system per test speech sequence
    # exp_per_seq(30, stride=1)
    # exp_per_seq(60, stride=1)
    # exp_per_seq(90, stride=1)
    # exp_per_seq(100, stride=1)

    '''
    output_2_new_wavlm 0.860,  184.753
    '''
