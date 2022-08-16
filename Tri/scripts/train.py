import pdb
import time
from pathlib import Path
import sys
import pprint

[sys.path.append(i) for i in ['.', '..']]
import torch
import matplotlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import speech2gesture, vocab
from model.embedding_net import EmbeddingNet
from train_eval.train_gan import train_iter_gan
from train_eval.train_joint_embed import train_iter_embed, eval_embed
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from config.parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils


args = parse_args()
device = torch.device("cuda:" + str(args.no_cuda[0]) if torch.cuda.is_available() else "cpu")

def init_model(args, lang_model, speaker_model, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = loss_fn = None
    if args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim, args).to(_device)
    elif args.model == 'speech2gesture':
        loss_fn = torch.nn.L1Loss()

    return generator, discriminator, loss_fn


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    # print_interval = int(len(train_data_loader) / 5)
    print_interval = 1
    save_sample_result_epoch_interval = 5
    save_model_epoch_interval = 5

    # z type
    if args.z_type == 'speaker':
        pass
    elif args.z_type == 'random':
        speaker_model = 1
    else:
        speaker_model = None

    # init model
    generator, discriminator, loss_fn = init_model(args, lang_model, speaker_model, pose_dim, device)

    # use multi GPUs
    print(torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator, device_ids=[eval(i) for i in args.no_cuda])
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator, device_ids=[eval(i) for i in args.no_cuda])



    # prepare an evaluator for FGD
    embed_space_evaluator = None
    # if args.eval_net_path and len(args.eval_net_path) > 0:
    #     embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.epochs):
        # '''
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args)
        
        
        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation', val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # '''
        # save model
        # is_best = True
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            # if is_best:
            #     save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            # else:
            #     save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                'dis_dict': dis_state_dict,
            }, save_name)

        '''
        # save sample results
        if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
            evaluate_sample_and_save_video(
                epoch, args.name + '_' + str(epoch), test_data_loader, generator,
                args=args, lang_model=lang_model)
        '''

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            _, _, in_text_padded, target_vec, in_audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data

            '''
            print(in_text, in_text.shape)       # tensor([0]) torch.Size([1]) in Trimodal
            tensor([[    1,    93,   158,  ...,   583,  1550,     2],
                    [    1,   621,    50,  ...,  7909,     2,     0],
                    [    1,  1033,   154,  ...,   324,     2,     0],
                    ...,
                    [    1,     6,   277,  ...,     0,     0,     0],
                    [    1, 17992,  5534,  ...,     0,     0,     0],
                    [    1,    20,     2,  ...,     0,     0,     0]]) torch.Size([128, 15])
            print(text_lengths, text_lengths.shape)     # tensor([0]) torch.Size([1])
            tensor([15, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                    12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10,
                    10, 10, 10, 10, 10, 10, 10,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
                     9,  9,  9,  9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
                     8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  7,  7,  7,  7,  7,  7,
                     7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,  6,  6,
                     6,  6,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,
                     4,  3]) torch.Size([128])
            print(in_text_padded, in_text_padded.shape)
            tensor([[   93,     0,   158,  ...,     0,  1550,     0],
                    [    0,   621,     0,  ...,  7909,     0,     0],
                    [    0,  1033,     0,  ...,    21,     0,   324],
                    ...,
                    [    6,     0,     0,  ...,     0,     0,     0],
                    [17992,     0,     0,  ...,     0,     0,     0],
                    [    0,     0,     0,  ...,     0,     0,     0]]) torch.Size([128, 34])
            print(target_vec, target_vec.shape)
            tensor([[[-3.6489e-02, -3.0709e-02,  9.8986e-02,  ...,  4.2556e-01,
                      -2.3333e-01,  1.5453e-01],
                    torch.Size([128, 34, 27])
            print(in_audio, in_audio.shape)
            tensor([[-0.0074, -0.0139, -0.0182,  ...,  0.0045,  0.0057,  0.0071],
                    [ 0.0040,  0.0008,  0.0017,  ...,  0.0225,  0.0361,  0.0328],
                    [-0.0002, -0.0003,  0.0005,  ...,  0.0027,  0.0031,  0.0025],
                    ...,
                    [ 0.0112,  0.0048,  0.0034,  ...,  0.1089,  0.0462, -0.0747],
                    [ 0.0190,  0.0143, -0.0294,  ..., -0.0042, -0.0040, -0.0037],
                    [ 0.0023,  0.0026,  0.0083,  ...,  0.1764,  0.1266,  0.0781]]) torch.Size([128, 36267]), 34/15*16000
            print(in_spec, in_spec.shape)
            tensor([[[-46.9062, -43.3438, -41.7500,  ..., -59.2188, -51.4062, -49.6250],
            torch.Size([128, 128, 70])
            print(aux_info)
            'vid',  'start_frame_no',  'end_frame_no', 'start_time', 'end_time', 'is_correct_motion', 'filtering_message'
            '''

            # print(pitch.shape, energy.shape, volume.shape, speech_emo.shape, text_emo.shape)
            # torch.Size([1, 151])
            # torch.Size([1, 151])
            # torch.Size([1, 151])
            # torch.Size([1, 8])
            # torch.Size([1, 3])

            # pdb.set_trace()


            batch_size = target_vec.size(0)

            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)

            pitch = pitch.type(torch.FloatTensor).to(device)
            energy = energy.type(torch.FloatTensor).to(device)
            volume = volume.type(torch.FloatTensor).to(device)

            if args.use_emo and args.use_txt_emo:
                text_emo = text_emo.type(torch.FloatTensor).to(device)
            if args.use_emo and args.use_audio_emo:
                speech_emo = speech_emo.type(torch.FloatTensor).to(device)


            # speaker input
            vid_indices = []
            if speaker_model and isinstance(speaker_model, vocab.Vocab):
                # print('speaker input')
                vids = aux_info['vid']
                vid_indices = [speaker_model.word2index[vid] for vid in vids]
                vid_indices = torch.LongTensor(vid_indices).to(device)
                # print(vid_indices.shape)        # torch.Size([128])

            # train
            loss = []
            if args.model == 'multimodal_context':
                loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_vec, vid_indices,
                                      generator, discriminator,
                                      gen_optimizer, dis_optimizer, pitch, energy, volume, speech_emo, text_emo)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

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

    tb_writer.close()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))


def evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args):
    # to evaluation mode
    generator.train(False)

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            _, _, in_text_padded, target_vec, in_audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data

            batch_size = target_vec.size(0)

            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            target = target_vec.to(device)
            pitch = pitch.type(torch.FloatTensor).to(device)
            energy = energy.type(torch.FloatTensor).to(device)
            volume = volume.type(torch.FloatTensor).to(device)
            if args.use_emo and args.use_txt_emo:
                text_emo = text_emo.type(torch.FloatTensor).to(device)
            if args.use_emo and args.use_audio_emo:
                speech_emo = speech_emo.type(torch.FloatTensor).to(device)

            # speaker input
            speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            # pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            if args.model == 'multimodal_context':
                # print(next(generator.parameters()).device)
                # print(pre_seq.device)
                # print(in_text_padded.device)
                # print(in_audio.device)
                # print(vid_indices.device)
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices, pitch, energy, volume, speech_emo, text_emo)
                loss = F.l1_loss(out_dir_vec, target)
                loss += F.mse_loss(out_dir_vec, target)
            else:
                assert False

            losses.update(loss.item(), batch_size)

            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                # calculate MAE of joint coordinates
                out_dir_vec = out_dir_vec.cpu().numpy()
                # out_dir_vec += np.array(args.data_mean).squeeze()
                out_joint_poses = out_dir_vec
                # out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                target_vec = target_vec.cpu().numpy()
                # target_vec += np.array(args.data_mean).squeeze()
                # target_poses = convert_dir_vec_to_pose(target_vec)
                target_poses = target_vec
                if out_joint_poses.shape[1] == args.n_poses:
                    diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                else:
                    diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        logging.info(
            '[VAL] loss: {:.3f}, joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                losses.avg, joint_mae.avg, accel.avg, frechet_dist, feat_dist, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
    else:
        logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s'.format(
            losses.avg, joint_mae.avg, elapsed_time))

    return ret_dict


from pymo.writers import BVHWriter
from scipy.signal import savgol_filter
import joblib as jl
from scipy.spatial.transform import Rotation as R
def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    pipeline_path = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/resource/data_pipe_18_20220624_1.sav>"
    pipeline = jl.load(pipeline_path)

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 12))  # (n_frames, n_joints, 12)
    out_data = np.zeros((out_poses.shape[0], out_poses.shape[1], 6))
    for i in range(out_poses.shape[0]):  # frames
        for j in range(out_poses.shape[1]):  # joints
            out_data[i, j, :3] = out_poses[i, j, :3]
            r = R.from_matrix(out_poses[i, j, 3:].reshape(3, 3))
            out_data[i, j, 3:] = r.as_euler('ZXY', degrees=True).flatten()

    out_data = out_data.reshape(out_data.shape[0], -1)
    bvh_data = pipeline.inverse_transform([out_data])

    out_bvh_path = os.path.join(save_path, filename_prefix + '_generated.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


def evaluate_sample_and_save_video(epoch, filename_prefix, test_data_loader, generator, args, lang_model,
                                   n_save=None, save_path=None):
    print('evaluate sample and save video')
    generator.train(False)  # eval mode
    start = time.time()
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    out_raw = []

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            if iter_idx >= n_save:  # save N samples
                break
            _, _, in_text_padded, target_dir_vec, in_audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data

            # prepare
            select_index = 0
            in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            target_dir_vec = target_dir_vec[select_index, :, :].unsqueeze(0).to(device)
            pitch = pitch[select_index, :].type(torch.FloatTensor).unsqueeze(0).to(device)
            energy = energy[select_index, :].type(torch.FloatTensor).unsqueeze(0).to(device)
            volume = volume[select_index, :].type(torch.FloatTensor).unsqueeze(0).to(device)
            if args.use_emo and args.use_audio_emo:
                speech_emo = speech_emo[select_index, :].type(torch.FloatTensor).unsqueeze(0).to(device)
            if args.use_emo and args.use_txt_emo:
                text_emo = text_emo[select_index, :].type(torch.FloatTensor).unsqueeze(0).to(device)

            input_words = []
            for i in range(in_text_padded.shape[1]):
                word_idx = int(in_text_padded.data[select_index, i])
                if word_idx > 0:
                    input_words.append(lang_model.index2word[word_idx])
            sentence = ' '.join(input_words)

            # speaker input
            speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid = aux_info['vid'][select_index]
                # vid_indices = [speaker_model.word2index[vid]]
                vid_indices = [random.choice(list(speaker_model.word2index.values()))]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            # aux info
            aux_str = '({}, time: {}-{})'.format(
                aux_info['vid'][select_index],
                str(datetime.timedelta(seconds=aux_info['start_time'][select_index].item())),
                str(datetime.timedelta(seconds=aux_info['end_time'][select_index].item())))

            # synthesize
            pre_seq = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1],
                                                target_dir_vec.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices, pitch, energy, volume, speech_emo, text_emo)

            # to video
            out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())

            # print(out_dir_vec.shape)

            if save_path is None:
                save_path = args.model_save_path

            make_bvh(save_path, filename_prefix, out_dir_vec)

    generator.train(True)  # back to training mode
    logging.info('saved sample videos, took {:.1f}s'.format(time.time() - start))

    return out_raw


def main(config):
    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

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

    print(len(train_loader), len(test_loader))

    # train
    pose_dim = args.pose_dim  # 18 x 3, 27 -> 54
    train_epochs(args, train_loader, test_loader, lang_model, pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    '''
    python train.py --config=<..your path/GENEA/genea_challenge_2022/baselines/Tri/config/multimodal_context.yml>
    '''

    _args = parse_args()
    main({'args': _args})
