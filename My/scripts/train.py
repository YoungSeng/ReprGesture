import datetime
import pprint
import random
import time
import sys
import numpy as np

from torch.utils.data import DataLoader

[sys.path.append(i) for i in ['.', '..']]

from model import vocab
from model.seq2seq_net import Seq2SeqNet
from train_eval.train_seq2seq import train_iter_seq2seq
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab

from config.parse_args import parse_args

from torch import optim

from twh_dataset_to_lmdb import target_joints
from data_loader.lmdb_data_loader import *
import utils.train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args, lang_model, pose_dim, _device):
    n_frames = args.n_poses
    generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                           lang_model.word_embedding_weights).to(_device)
    loss_fn = torch.nn.MSELoss()

    return generator, loss_fn


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, trial_id=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss')]

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 20

    # init model
    generator, loss_fn = init_model(args, lang_model, pose_dim, device)

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    for epoch in range(1, args.epochs+1):
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, args)

        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0:
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, target_vec, in_audio, aux_info = data


            '''
            print(in_text, in_text.shape)
            tensor([[  1,  15,  11,  ...,  57,  20,   2],
                    [  1,  27,  43,  ...,  62, 377,   2],
                    [  1, 188,  30,  ...,   2,   0,   0],
                    ...,
                    [  1, 104,  32,  ...,   0,   0,   0],
                    [  1, 154,  10,  ...,   0,   0,   0],
                    [  1,  43, 115,  ...,   0,   0,   0]]) torch.Size([128, 13]), (batch, len)
            print(text_lengths, text_lengths.shape)
            tensor([13, 13, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  9,  9,
                     9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  8,  8,
                     8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                     8,  8,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
                     7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,  6,  6,  6,
                     6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
                     5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
                     5,  5]) torch.Size([128]), (batch)
            print(target_vec, target_vec.shape)
            tensor([[[ 0.0000e+00,  4.3750e-04, -1.8750e-04,  ..., -1.0099e+00,
                      -5.6970e-02,  1.0419e+00],
                     [ 0.0000e+00,  4.3750e-04, -1.8750e-04,  ..., -1.4105e+00,
                       4.2096e-01,  5.6595e-01],
                     [ 0.0000e+00,  4.3750e-04, -1.8750e-04,  ..., -1.4730e+00,
                       4.9918e-01,  4.5106e-01],
                     ...,
                     torch.Size([128, 40, 216]), (batch, 40, 216)
            print(in_audio, in_audio.shape)
            tensor([[-0.0426, -0.0423, -0.0420,  ..., -0.0446, -0.0373, -0.0469],
                    [-0.0702, -0.0738, -0.0757,  ...,  0.0033,  0.0087,  0.0126],
                    [-0.0072, -0.0102, -0.0120,  ..., -0.0495, -0.0510, -0.0524],
                    ...,
                    [ 0.1551,  0.1554,  0.1558,  ...,  0.1724,  0.1674,  0.1743],
                    [-0.0125, -0.0095, -0.0052,  ..., -0.0409, -0.0392, -0.0385],
                    [-0.1625, -0.1634, -0.1580,  ..., -0.1248, -0.1241, -0.1217]]) torch.Size([128, 21333]), (batch, 21333), 40/30*16000
            print(aux_info)
            {'vid': ['val_2022_v1_006', ...'start_frame_no': tensor([1210, ... 'end_frame_no': tensor([1250,...'start_time': tensor([40.3333,... 'end_time': tensor([41.6667, 20.6667,...
            '''

            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)

            # train
            loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()


def evaluate_testset(test_data_loader, generator, loss_fn, args):
    # to evaluation mode
    generator.train(False)

    losses = AverageMeter('loss')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, target_vec, in_audio, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            target = target_vec.to(device)

            out_poses = generator(in_text, text_lengths, target, None)
            loss = loss_fn(out_poses, target)
            losses.update(loss.item(), batch_size)

    # back to training mode
    generator.train(True)

    # print
    elapsed_time = time.time() - start
    logging.info('[VAL] loss: {:.3f} / {:.1f}s'.format(losses.avg, elapsed_time))

    return losses.avg


def main(config):
    args = config['args']

    trial_id = None

    # random seed
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info(pprint.pformat(vars(args)))

    # dataset
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

    # train
    train_epochs(args, train_loader, test_loader, lang_model,
                 pose_dim=len(target_joints)*12, trial_id=trial_id)     # 18*12=216


if __name__ == '__main__':
    '''
    python train.py --config="<..your path/GENEA/genea_challenge_2022/baselines/My/config/seq2seq.yml>"
    '''
    _args = parse_args()
    main({'args': _args})
