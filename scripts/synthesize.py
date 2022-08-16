import argparse
import math
import pdb
import pickle
import pprint
import subprocess
import time
import os
import numpy as np
import torch
import joblib as jl

from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

import utils
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger
from data_loader.data_preprocessor import DataPreprocessor

from pymo.writers import BVHWriter
import random
from energy import AudioProcesser

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")



def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentence_to_emotion(words):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from scipy.special import softmax
    MODEL = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/cache_text/>"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model_ = AutoModelForSequenceClassification.from_pretrained(MODEL)
    n_sentence = ' '.join([i[0] for i in words])
    text = preprocess(n_sentence)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model_(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores


def main(checkpoint_path, transcript_path, wav_path, vid=None):
    args, generator, loss_fn, lang_model, speaker_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path, device)
    pprint.pprint(vars(args))

    # vid = random.sample(range(0, speaker_model.n_words), 1)[0]  # for trimodal

    save_path = '../output/infer_sample/output_2_MISA_48_nodif/bvh'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    root = "<..your path/GENEA/genea_challenge_2022/dataset/v1_18/val/>"
    tsv_path = root + 'tsv/'
    wave_path = root + 'wav/'
    tmp_list = os.listdir(tsv_path)

    for item in tmp_list:
        transcript_path = tsv_path + item
        wav_path = wave_path + item[:-4] + '.wav'



        # prepare input
        transcript = SubtitleWrapper(transcript_path).get()
        word_list = []
        for wi in range(len(transcript)):
            word_s = float(transcript[wi][0])
            word_e = float(transcript[wi][1])
            word = transcript[wi][2].strip()

            word_tokens = word.split()

            for t_i, token in enumerate(word_tokens):
                token = normalize_string(token)
                if len(token) > 0:
                    new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    word_list.append([token, new_s_time, new_e_time])

        # inference
        import librosa
        audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
        ap = AudioProcesser(wav_path, hop_size=128)  # 320 = 20ms, 16000/hop_size = 50
        energy = ap.get_energy()
        pitch = ap.get_pitch(log=True, norm=False)
        volume = ap.calVolume()

        out_list = []
        n_frames = args.n_poses
        clip_length = len(audio_raw) / audio_sr
        print(clip_length)
        pre_seq = torch.zeros((1, n_frames, len(args.data_mean) + 1))       # 20220627

        # mean_pose = args.data_mean
        # mean_pose = torch.squeeze(torch.Tensor(mean_pose))      # (216)
        # pre_seq[0, :, :] = mean_pose.repeat(args.n_poses, 1)
        # pre_seq[0, 0:args.n_pre_poses, :-1] = mean_pose[0:args.n_pre_poses]
        # pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

        unit_time = args.n_poses / args.motion_resampling_framerate
        stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
        if clip_length < unit_time:
            num_subdivision = 1
        else:
            num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
        audio_sample_length = int(unit_time * audio_sr)
        pitch_sample_length = int(unit_time * 16000 * 7876 / 2778300)
        end_padding_duration = 0
        # prepare speaker input
        if args.z_type == 'speaker':
            if not vid:
                vid = random.randrange(generator.z_obj.n_words)
            print('vid:', vid)
            vid = torch.LongTensor([vid]).to(device)
        else:
            vid = None
        print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

        out_dir_vec = None
        start = time.time()
        for i in range(0, num_subdivision):
            start_time = i * stride_time
            end_time = start_time + unit_time
            audio_start = math.floor(start_time / clip_length * len(audio_raw))
            audio_end = audio_start + audio_sample_length
            in_audio = audio_raw[audio_start:audio_end]

            pitch_start = math.floor(start_time / clip_length * len(pitch))
            pitch_end = pitch_start + pitch_sample_length
            in_pitch = pitch[pitch_start:pitch_end]

            in_energy = energy[pitch_start:pitch_end]

            in_volume = volume[pitch_start:pitch_end]

            if len(in_audio) < audio_sample_length:
                if i == num_subdivision - 1:
                    end_padding_duration = audio_sample_length - len(in_audio)
                in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
                in_pitch = np.pad(in_pitch, (0, pitch_sample_length - len(in_pitch)), 'constant')
                in_energy = np.pad(in_energy, (0, pitch_sample_length - len(in_energy)), 'constant')
                in_volume = np.pad(np.squeeze(in_volume), (0, pitch_sample_length - len(in_volume)), 'constant')
            in_audio = torch.as_tensor(in_audio).unsqueeze(0).to(device).float()

            in_energy = torch.as_tensor(in_energy).unsqueeze(0).to(device).float()
            in_pitch = torch.as_tensor(in_pitch).unsqueeze(0).to(device).float()
            in_volume = torch.as_tensor(in_volume).squeeze().unsqueeze(0).to(device).float()

            # prepare text input
            word_seq = DataPreprocessor.get_words_in_time_range(word_list=word_list, start_time=start_time, end_time=end_time)
            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            word_indices = np.zeros(len(word_seq) + 2)
            word_indices[0] = lang_model.SOS_token
            word_indices[-1] = lang_model.EOS_token
            frame_duration = (end_time - start_time) / n_frames
            for w_i, word in enumerate(word_seq):
                print(word[0], end=', ')
                idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
                extended_word_indices[idx] = lang_model.get_word_index(word[0])
                word_indices[w_i + 1] = lang_model.get_word_index(word[0])
            print(' ')
            in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)

            # prepare pre seq
            if i > 0:
                pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
                pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq = pre_seq.float().to(device)

            if not args.use_emo:
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid, in_pitch, in_energy, in_volume, None, None)

            elif args.use_txt_emo and not args.use_audio_emo:
                text_emo = torch.as_tensor(np.array(sentence_to_emotion(word_seq))).unsqueeze(0).to(device).float()
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid, in_pitch, in_energy, in_volume, None, text_emo)



            out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

            # smoothing motion transition
            if len(out_list) > 0:
                last_poses = out_list[-1][-args.n_pre_poses:]
                out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 10 frames

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[j]
                    next = out_seq[j]
                    out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)


            out_list.append(out_seq)

        print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))


        # aggregate results
        out_poses = np.vstack(out_list)
        # out_poses = out_poses[:-args.n_pre_poses]       # 20220628
        # out_poses = out_poses[args.n_pre_poses:]  # 20220628

        # unnormalize
        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        out_poses = np.multiply(out_poses, std) + mean

        # make a BVH
        filename_prefix = transcript_path[:-4].split('/')[-1]
        print(filename_prefix)
        make_bvh(save_path, filename_prefix, out_poses)


def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    pipeline = jl.load("<..your path/GENEA/genea_challenge_2022/baselines/Tri/resource/data_pipe_18_20220624_1.sav>")

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

    out_bvh_path = os.path.join(save_path, filename_prefix + '.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


if __name__ == '__main__':
    '''
    python synthesize.py --ckpt_path "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output_2_MISA_48_nodif/train_multimodal_context/multimodal_context_checkpoint_080.bin>" --transcript_path "<..your path/GENEA/genea_challenge_2022/dataset/v1_18/val/tsv/val_2022_v1_000.tsv>" --wav_path "<..your path/GENEA/genea_challenge_2022/dataset/v1_18/val/wav/val_2022_v1_000.wav>"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--transcript_path", type=str)
    parser.add_argument("--wav_path", type=str)
    args = parser.parse_args()

    main(args.ckpt_path, args.transcript_path, args.wav_path)





