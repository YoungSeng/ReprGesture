import datetime
import logging
import os
import pdb
import pickle
import random

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,2,0'

import numpy as np
import lmdb as lmdb
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import sys
[sys.path.append(i) for i in ['.', '..']]

import utils.train_utils
import utils.data_utils
from model.vocab import Vocab
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow
import json
import requests
import librosa

from src.models import Wav2Vec2ForSpeechClassification
import torch
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import torch.nn.functional as F
import torchaudio
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, extended_word_seq, poses_seq, audio, aux_info, pitch, energy, volume, speech_emo, text_emo = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    extended_word_seq = default_collate(extended_word_seq)
    poses_seq = default_collate(poses_seq)
    audio = default_collate(audio)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}
    pitch = default_collate(pitch)
    energy = default_collate(energy)
    volume = default_collate(volume)
    if args.use_emo and args.use_audio_emo:
        speech_emo = default_collate(speech_emo)
    if args.use_emo and args.use_txt_emo:
        text_emo = default_collate(text_emo)

    return word_seq, words_lengths, extended_word_seq, poses_seq, audio, aux_info, pitch, energy, volume, speech_emo, text_emo


from config.parse_args import parse_args
args = parse_args()

if args.use_emo:
    if args.use_audio_emo:
        model_name_or_path = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/cache/>"
        config = AutoConfig.from_pretrained(model_name_or_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        sampling_rate = feature_extractor.sampling_rate
        # torch.multiprocessing.set_start_method('forkserver', force=True)
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path)
    if args.use_txt_emo:
        MODEL = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/cache_text/>"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model_ = AutoModelForSequenceClassification.from_pretrained(MODEL)



class TwhDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std, speaker_model=None):

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

        # make a speaker model
        if speaker_model is None or speaker_model == 0:
            precomputed_model = lmdb_dir + '_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                print('make_speaker_model')
                self._make_speaker_model(lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model

        API = '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'
        self.headers = {"Authorization": f"Bearer " + API}
        self.API_URL_sentence = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.API_URL_speech = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info, pitch, energy, volume = sample


        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            prev_idx = 0
            for word in words:
                idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                if idx < n_frames:
                    extended_word_indices[idx] = lang.get_word_index(word[0])
                    # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                    prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        def preprocess(text):
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            return " ".join(new_text)

        def sentence_to_emotion(words):
            n_sentence = ' '.join([i[0] for i in words])
            text = preprocess(n_sentence)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model_(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            return scores



        '''     # API
        def sentence_to_emotion(words):
            n_sentence = ' '.join([i[0] for i in words])
            data = json.dumps({"inputs": n_sentence})
            response = requests.request("POST", self.API_URL_sentence, headers=self.headers, data=data)
            return [i['score'] for i in json.loads(response.content.decode("utf-8"))[0]]        # 'Negative', 'Neutral', 'Positive'
        '''
        '''     # API
        def speech_to_emotion(tmp_audio):
            # print(tmp_audio.shape)
            tmp_path = '<..your path/GENEA/genea_challenge_2022/baselines/Tri/temp.wav>'
            librosa.output.write_wav(tmp_path, tmp_audio, 16000, norm=False)
            with open(tmp_path, "rb") as f:
                data_ = f.read()
            response = requests.request("POST", self.API_URL_speech, headers=self.headers, data=data_)
            data = json.loads(response.content.decode("utf-8"))
            if 'error' in data:
                print(data['error'])
                return [0.125 for _ in range(8)]

            emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
            result = [0 for _ in range(8)]
            for item in data:
                result[emotions.index(item['label'])] = item['score']
            pad = (1 - sum(result)) / 3
            result_ = [pad if i == 0 else i for i in result]
            return result_
        '''


        def speech_file_to_array_fn(path, sampling_rate):
            speech_array, _sampling_rate = torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            return speech

        def predict(path, sampling_rate=16000):
            speech = speech_file_to_array_fn(path, sampling_rate)
            inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {key: inputs[key] for key in inputs}
            with torch.no_grad():
                logits = model(**inputs).logits
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
                       enumerate(scores)]
            return outputs

        def speech_to_emotion(tmp_audio):
            tmp_path = '<..your path/GENEA/genea_challenge_2022/baselines/Tri/temp.wav>'
            librosa.output.write_wav(tmp_path, tmp_audio, 16000, norm=False)
            try:
                outputs = predict(tmp_path)
                # print(outputs)
            except:
                return [0.125 for _ in range(8)]
            # outputs = predict(tmp_path)
            # print(outputs)
            emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
            result = [0 for _ in range(8)]
            for item in outputs:
                result[emotions.index(item['Emotion'])] = eval(item['Score'][:-1]) / 100
            pad = (1 - sum(result)) / 3
            result_ = [pad if i == 0 else i for i in result]
            return result_

        volume = volume.squeeze()

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std

        if args.use_emo:
            if args.use_txt_emo and args.use_audio_emo:
                text_emo = np.array(sentence_to_emotion(word_seq))
                speech_emo = np.array(speech_to_emotion(audio))
            elif args.use_txt_emo:
                text_emo = np.array(sentence_to_emotion(word_seq))
                speech_emo = None
            elif args.use_audio_emo:
                speech_emo = np.array(speech_to_emotion(audio))
                text_emo = None
        else:
            speech_emo = None
            text_emo = None

        # print(' '.join([i[0] for i in word_seq]), text_emo)
        # print(speech_emo)

        '''
        yeah yeah who cares that is amazing [0.012040417641401291, 0.027290182188153267, 0.9606693387031555]
        okay yeah yeah who cares that is amazing [0.01274406723678112, 0.027020933106541634, 0.9602349996566772]
        yeah yeah it is okay it literally looks like it is [0.005745251197367907, 0.07453840225934982, 0.9197163581848145]
        yeah yeah that is probably not great [0.8941145539283752, 0.09355661273002625, 0.012328829616308212]
        '''

        # to tensors
        if self.lang_model:
            word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
            extended_word_seq = extend_word_seq(self.lang_model, word_seq, aux_info['end_time'])
        else:
            word_seq_tensor = 0
            extended_word_seq = 0

        pose_seq = torch.from_numpy(np.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(np.copy(audio)).float()

        return word_seq_tensor, extended_word_seq, pose_seq, audio, aux_info, pitch, energy, volume, speech_emo, text_emo      # word_seq

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model


    def _make_speaker_model(self, lmdb_dir, cache_path):
        logging.info('  building a speaker model...')
        speaker_model = Vocab('vid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            speaker_model.index_word(vid)

        lmdb_env.close()
        logging.info('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)


if __name__ == '__main__':
    '''
    python data_loader/lmdb_data_loader.py --config <..your path/GENEA/genea_challenge_2022/baselines/Tri/config/multimodal_context.yml>
    '''
    from torch.utils.data import DataLoader
    from utils.vocab_utils import build_vocab
    from config.parse_args import parse_args


    model_name_or_path = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/cache/>"
    config = AutoConfig.from_pretrained(model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    sampling_rate = feature_extractor.sampling_rate
    # torch.multiprocessing.set_start_method('forkserver', force=True)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path)

    MODEL = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/cache_text/>"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model_ = AutoModelForSequenceClassification.from_pretrained(MODEL)

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

    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    flag = 2
    for iter_idx, data in enumerate(train_loader, 0):
        print(iter_idx)
        word_seq, words_lengths, extended_word_seq, poses_seq, audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data

        print(word_seq, word_seq.shape)
        print(words_lengths, words_lengths.shape)
        print(extended_word_seq, extended_word_seq.shape)       # torch.Size([128, 100])
        print(poses_seq, poses_seq.shape)       # torch.Size([128, 100, 216])
        print(audio, audio.shape)       # torch.Size([128, 53333])
        print(aux_info)
        print(pitch, pitch.shape)       # torch.Size([8, 150])
        print(energy, energy.shape)  # torch.Size([8, 150])
        print(volume, volume.shape)  # torch.Size([8, 151])
        print(speech_emo, speech_emo.shape)  # torch.Size([8, 8])
        print(text_emo, text_emo.shape)  # torch.Size([8, 3])

        '''
        tensor([[0.1304, 0.1239, 0.1267, 0.1178, 0.1404, 0.1178, 0.1253, 0.1178],
        [0.1125, 0.1125, 0.1322, 0.1382, 0.1217, 0.1293, 0.1411, 0.1125]],
       dtype=torch.float64) torch.Size([2, 8])
        tensor([[0.8941, 0.0936, 0.0123],
                [0.0191, 0.1179, 0.8630]], dtype=torch.float64) torch.Size([2, 3])

        '''

        if iter_idx == flag:
            break





