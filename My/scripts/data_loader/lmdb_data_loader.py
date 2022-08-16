import logging
import os

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import sys
[sys.path.append(i) for i in ['.', '..']]

from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, poses_seq, audio, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    poses_seq = default_collate(poses_seq)
    audio = default_collate(audio)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, poses_seq, audio, aux_info


class TwhDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info = sample

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std

        # to tensors
        if self.lang_model:
            word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
        else:
            word_seq_tensor = 0

        pose_seq = torch.from_numpy(np.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(np.copy(audio)).float()

        return word_seq_tensor, pose_seq, audio, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model


if __name__ == '__main__':


    '''
    python data_loader/lmdb_data_loader.py --config="<..your path/GENEA/genea_challenge_2022/baselines/My/config/seq2seq.yml>"
    '''

    from torch.utils.data import DataLoader
    from config.parse_args import parse_args
    from utils.vocab_utils import build_vocab

    args = parse_args()

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

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    flag = 2
    for iter_idx, data in enumerate(train_loader, 0):
        print(iter_idx)
        in_text, text_lengths, target_vec, in_audio, aux_info = data

        print(in_text, in_text.shape)
        print(text_lengths, text_lengths.shape)
        # print(in_text_padded, in_text_padded.shape)
        print(target_vec, target_vec.shape)
        print(in_audio, in_audio.shape)
        # print(in_spec, in_spec.shape)
        print(aux_info)

        if iter_idx == flag:
            break
