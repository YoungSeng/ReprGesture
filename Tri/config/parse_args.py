import configargparse
# from pathlib import Path

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


import torch.nn as nn
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh, 'my': nn.LeakyReLU(inplace=True)}


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=False, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main")
    parser.add("--train_data_path", action="append")
    parser.add("--val_data_path", action="append")
    parser.add("--test_data_path", action="append")
    parser.add("--model_save_path", required=False)
    parser.add("--pose_representation", type=str, default='3d_vec')
    # parser.add("--mean_dir_vec", action="append", type=float, nargs='*')
    # parser.add("--mean_pose", action="append", type=float, nargs='*')
    parser.add("--data_mean", action="append", type=float, nargs='*')
    parser.add("--data_std", action="append", type=float, nargs='*')
    parser.add("--random_seed", type=int, default=-1)
    parser.add("--save_result_video", type=str2bool, default=True)

    # word embedding
    parser.add("--wordembed_path", type=str, default=None)
    parser.add("--wordembed_dim", type=int, default=100)
    parser.add("--freeze_wordembed", type=str2bool, default=False)

    # model
    parser.add("--model", type=str, default='multimodal_context')
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--n_layers", type=int, default=2)
    parser.add("--hidden_size", type=int, default=200)
    parser.add("--z_type", type=str, default='none')
    parser.add("--input_context", type=str, default='both')

    # dataset
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=100)
    parser.add("--n_pre_poses", type=int, default=10)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=0)

    # GAN parameter
    parser.add("--GAN_noise_size", type=int, default=0)

    # training
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--discriminator_lr_weight", type=float, default=0.2)
    parser.add("--loss_regression_weight", type=float, default=50)
    parser.add("--loss_gan_weight", type=float, default=1.0)
    parser.add("--loss_kld_weight", type=float, default=0.1)
    parser.add("--loss_reg_weight", type=float, default=0.01)
    parser.add("--loss_warmup", type=int, default=-1)

    # eval
    parser.add("--eval_net_path", type=str, default='')

    # 20220620
    parser.add("--dim_audio", type=int, default=128)
    parser.add("--dim_text", type=int, default=32)
    parser.add("--dim_video", type=int, default=18*12+1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', default=activation_dict['my'])
    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--use_cmd_sim', type=str2bool, default=False)       # 20220621
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    parser.add_argument('--size_space', type=int, default=48)       # 20220701 128 -> 32
    parser.add_argument('--pose_dim', type=int, default=18*12)
    parser.add("--no_cuda", type=list, default=['2','0','1','3'])

    parser.add_argument('--use_emo', type=str2bool, default=False)  # 20220624
    parser.add_argument('--use_wavlm', type=str2bool, default=True)  # 20220624

    parser.add_argument('--use_txt_emo', type=str2bool, default=False)  # 20220624
    parser.add_argument('--use_audio_emo', type=str2bool, default=False)  # 20220624

    parser.add_argument('--ckpt_path', type=str, default="<..your path/GENEA/genea_challenge_2022/baselines/Tri/output_2/train_multimodal_context/multimodal_context_checkpoint_010.bin>")  # 20220624
    parser.add_argument('--transcript_path', type=str, default="<..your path/GENEA/genea_challenge_2022/dataset/v1_18/val/tsv/val_2022_v1_000.tsv>")  # 20220624
    parser.add_argument('--wav_path', type=str, default="<..your path/GENEA/genea_challenge_2022/dataset/v1_18/val/wav/val_2022_v1_000.wav>")  # 20220624

    parser.add_argument('--use_MISA', type=str2bool, default=True)  # 20220627

    parser.add_argument('--use_diff', type=str2bool, default=False)  # 20220627

    parser.add_argument('--use_reconstruct', type=str2bool, default=True)  # 20220815

    args = parser.parse_args()
    return args
