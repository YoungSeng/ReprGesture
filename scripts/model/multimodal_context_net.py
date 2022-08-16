import torch
import torch.nn as nn

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
[sys.path.append(i) for i in ['.', '..']]

from model import vocab
import model.embedding_net
from model.tcn import TemporalConvNet
import pdb
from mutimodal.models import MISA
from mutimodal.utils import CMD, DiffLoss, MSE
from config.parse_args import parse_args
args = parse_args()
""" https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html """

device = torch.device("cuda:" + str(args.no_cuda[0]) if torch.cuda.is_available() else "cpu")

if args.use_wavlm:
    from wavlm.WavLM import WavLM, WavLMConfig
    class WavEncoder(nn.Module):        # (batch, 166, 1024)
        def __init__(self, args):
            super().__init__()
            self.conv_reduce = nn.Sequential(
                nn.Conv1d(1024, 512, 23),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(512, 256, 23),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(256, 128, 23),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.3, inplace=True),
            )

            # load the pre-trained checkpoints
            path = "<..your path/GENEA/genea_challenge_2022/baselines/multimodal-deep-learning/My_MISA/wavlm_cache/WavLM-Large.pt>"
            checkpoint = torch.load(path)
            cfg = WavLMConfig(checkpoint['cfg'])
            self.feat_extractor = WavLM(cfg)
            self.feat_extractor.load_state_dict(checkpoint['model'])
            self.feat_extractor.eval()

            for name, param in self.feat_extractor.named_parameters():
                param.requires_grad = False

        def forward(self, wav_input_16khz):
            # extract the representation of last layer
            rep = self.feat_extractor.extract_features(wav_input_16khz)[0]
            x = self.conv_reduce(rep.permute(0, 2, 1))
            return x.permute(0, 2, 1)
            # return torch.rand(1, 100, 128).cuda(torch.device('cuda:3'))
else:
    class WavEncoder(nn.Module):  # (batch, 166, 1024)
        def __init__(self, args):
            super().__init__()
            self.feat_extractor = nn.Sequential(
                nn.Conv1d(1, 16, 25, stride=5, padding=200),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(16, 32, 25, stride=5),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(32, 64, 25, stride=5),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(64, 128, 25, stride=4),
            )

        def forward(self, wav_input_16khz):
            rep = self.feat_extractor(wav_input_16khz.unsqueeze(1))
            return rep.permute(0, 2, 1)

class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


if args.use_MISA:
    class PoseGenerator(nn.Module):
        def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings, z_obj=None):
            super().__init__()
            self.pre_length = args.n_pre_poses
            self.gen_length = args.n_poses - args.n_pre_poses
            self.z_obj = z_obj
            self.input_context = args.input_context

            if self.input_context == 'both':
                self.in_size = args.dim_audio + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
            elif self.input_context == 'none':
                self.in_size = pose_dim + 1
            else:
                self.in_size = 32 + pose_dim + 1  # audio or text only

            self.audio_encoder = WavEncoder(args)
            self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                               dropout=args.dropout_prob)

            self.speaker_embedding = None
            if self.z_obj:
                self.z_size = 16
                self.in_size += self.z_size
                if isinstance(self.z_obj, vocab.Vocab):
                    self.speaker_embedding = nn.Sequential(
                        nn.Embedding(z_obj.n_words, self.z_size),
                        nn.Linear(self.z_size, self.z_size)
                    )
                    self.speaker_mu = nn.Linear(self.z_size, self.z_size)
                    self.speaker_logvar = nn.Linear(self.z_size, self.z_size)
                else:
                    pass  # random noise

            self.hidden_size = args.hidden_size
            self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layers, batch_first=True,
                              bidirectional=True, dropout=args.dropout_prob)
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.hidden_size//2, pose_dim)
            )

            self.do_flatten_parameters = False
            if torch.cuda.device_count() > 1:
                self.do_flatten_parameters = True

            self.mutimodal_fusiom = MISA(args)

            self.con1d_pitch = nn.Sequential(
                nn.Conv1d(1, 2, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(2),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(2, 4, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(4, 8, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),
            )

            self.con1d_erengy = nn.Sequential(
                nn.Conv1d(1, 2, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(2),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(2, 4, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(4, 8, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),
            )

            self.con1d_volume = nn.Sequential(
                nn.Conv1d(1, 2, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(2),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(2, 4, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(4, 8, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),
            )

            encoder_layer_1 = nn.TransformerEncoderLayer(d_model=args.size_space * 3 + 8 * 3, nhead=2)
            self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1)

            self.fusion_1 = nn.Sequential()
            self.fusion_1.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 3 + 8 * 3, out_features=args.size_space * 3))
            self.fusion_1.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
            self.fusion_1.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 3, out_features=args.size_space * 2))  # self.config.pose_dim



            if args.use_emo and args.use_txt_emo and args.use_audio_emo:
                encoder_layer_2 = nn.TransformerEncoderLayer(d_model=args.size_space * 2 + 8 + 3, nhead=3)  # 需要整除
                self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
                self.fusion_2 = nn.Sequential()
                self.fusion_2.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 2 + 8 + 3, out_features=args.size_space * 2))
                self.fusion_2.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_2.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2, out_features=args.pose_dim))  # self.config.pose_dim

            elif args.use_emo and args.use_txt_emo:
                encoder_layer_2 = nn.TransformerEncoderLayer(d_model=args.size_space * 2 + 3, nhead=3)  # 需要整除
                self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
                self.fusion_txt_emo = nn.Sequential()
                self.fusion_txt_emo.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 2 + 3,
                                                                           out_features=args.size_space * 2))
                self.fusion_txt_emo.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_txt_emo.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2,
                                                                           out_features=args.pose_dim))  # self.config.pose_dim
            elif args.use_emo and args.use_audio_emo:
                encoder_layer_2 = nn.TransformerEncoderLayer(d_model=args.size_space * 2 + 8, nhead=2)  # 需要整除
                self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
                self.fusion_audio_emo = nn.Sequential()
                self.fusion_audio_emo.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 2 + 8,
                                                                           out_features=args.size_space * 2))
                self.fusion_audio_emo.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_audio_emo.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2,
                                                                           out_features=args.pose_dim))  # self.config.pose_dim
            else:
                self.fusion_wo_emo = nn.Sequential()
                self.fusion_wo_emo.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 3 + 8 * 3, out_features=args.size_space * 2))
                self.fusion_wo_emo.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_wo_emo.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2, out_features=args.pose_dim))  # self.config.pose_dim



            self.args = args


        def forward(self, pre_seq, in_text, in_audio, vid_indices=None, pitch=None, energy=None, volume=None, speech_emo=None, text_emo=None):
            '''
            # print(pre_seq.shape)        # torch.Size([128, 40, 217])
            # print(in_text.shape)        # torch.Size([128, 40])
            # print(in_audio.shape)       # torch.Size([128, 21333])
            '''

            decoder_hidden = None
            if self.do_flatten_parameters:
                self.gru.flatten_parameters()

            text_feat_seq = audio_feat_seq = None
            if self.input_context != 'none':
                # audio
                audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)

                # text
                text_feat_seq, _ = self.text_encoder(in_text)       # torch.Size([128, 40, 32])
                assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1])

            # z vector; speaker embedding or random noise
            if self.z_obj:
                if self.speaker_embedding:
                    assert vid_indices is not None
                    z_context = self.speaker_embedding(vid_indices)     # -> (128, 16)
                    z_mu = self.speaker_mu(z_context)       # -> (128, 16)
                    z_logvar = self.speaker_logvar(z_context)       # -> (128, 16)
                    z_context = model.embedding_net.reparameterize(z_mu, z_logvar)
                else:
                    z_mu = z_logvar = None
                    z_context = torch.randn(in_text.shape[0], self.z_size, device=in_text.device)
            else:
                z_mu = z_logvar = None
                z_context = None

            if self.input_context == 'both':
                # in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)
                decoder_outputs = self.mutimodal_fusiom(sentences=text_feat_seq, video=pre_seq, acoustic=audio_feat_seq)
                length = pre_seq.size()[1]
                energy_ = self.con1d_erengy(energy.unsqueeze(1)).permute(0, 2, 1)
                pitch_ = self.con1d_erengy(pitch.unsqueeze(1)).permute(0, 2, 1)
                volume_ = self.con1d_volume(volume.unsqueeze(1)).permute(0, 2, 1)

                h1_ = torch.cat((decoder_outputs, energy_, pitch_, volume_), dim=2)

                h1 = self.transformer_encoder_1(h1_)

                if args.use_emo and args.use_txt_emo and args.use_audio_emo:
                    h = self.fusion_1(h1)       # torch.Size([1, 100, 256])
                    speech_emo_ = speech_emo.unsqueeze(1).repeat(1, length, 1)
                    text_emo_ = text_emo.unsqueeze(1).repeat(1, length, 1)
                    h2_ = torch.cat((h, speech_emo_, text_emo_), dim=2)
                    h2 = self.transformer_encoder_2(h2_)
                    output = self.fusion_2(h2)
                elif args.use_emo and args.use_txt_emo:
                    h = self.fusion_1(h1)  # torch.Size([1, 100, 256])
                    text_emo_ = text_emo.unsqueeze(1).repeat(1, length, 1)
                    h2_ = torch.cat((h, text_emo_), dim=2)
                    h2 = self.transformer_encoder_2(h2_)
                    output = self.fusion_txt_emo(h2)
                elif args.use_emo and args.use_audio_emo:
                    h = self.fusion_1(h1)  # torch.Size([1, 100, 256])
                    speech_emo_ = speech_emo.unsqueeze(1).repeat(1, length, 1)
                    h2_ = torch.cat((h, speech_emo_), dim=2)
                    h2 = self.transformer_encoder_2(h2_)
                    output = self.fusion_audio_emo(h2)
                else:
                    output = self.fusion_wo_emo(h1)

            elif self.input_context == 'audio':
                in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
            elif self.input_context == 'text':
                in_data = torch.cat((pre_seq, text_feat_seq), dim=2)
            elif self.input_context == 'none':
                in_data = pre_seq
            else:
                assert False

            if z_context is not None:
                repeated_z = z_context.unsqueeze(1)
                repeated_z = repeated_z.repeat(1, in_data.shape[1], 1)
                in_data = torch.cat((in_data, repeated_z), dim=2)

            # # print(in_data.shape)        # torch.Size([128, 40, 281])
            # output, decoder_hidden = self.gru(in_data, decoder_hidden)
            # # print(output.shape)       # torch.Size([128, 40, 600])
            # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs, 128 40 300
            # output = self.out(output.reshape(-1, output.shape[2]))      # 128*40, 300->217
            # decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)

            # pdb.set_trace()
            # return output, z_context, z_mu, z_logvar
            if args.use_cmd_sim:
                return output, z_context, z_mu, z_logvar, self.get_cmd_loss(), self.get_diff_loss(), self.get_recon_loss()
            else:
                return output, z_context, z_mu, z_logvar, self.get_domain_loss(), self.get_diff_loss(), self.get_recon_loss()

        def get_cmd_loss(self):
            # if not self.args.use_cmd_sim:
            #     return 0.0
            self.loss_cmd = CMD()
            # losses between shared states
            loss = self.loss_cmd(self.mutimodal_fusiom.utt_shared_t, self.mutimodal_fusiom.utt_shared_v, 5)
            loss += self.loss_cmd(self.mutimodal_fusiom.utt_shared_t, self.mutimodal_fusiom.utt_shared_a, 5)
            loss += self.loss_cmd(self.mutimodal_fusiom.utt_shared_a, self.mutimodal_fusiom.utt_shared_v, 5)
            loss = loss/3.0

            return loss

        def get_domain_loss(self):

            # if self.args.use_cmd_sim:
            #     return 0.0
            self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
            # Predicted domain labels
            dim = self.mutimodal_fusiom.domain_label_t.size(2)
            domain_pred_t = self.mutimodal_fusiom.domain_label_t.view(-1, dim).to(device)
            domain_pred_v = self.mutimodal_fusiom.domain_label_v.view(-1, dim).to(device)
            domain_pred_a = self.mutimodal_fusiom.domain_label_a.view(-1, dim).to(device)

            # True domain labels
            domain_true_t = torch.LongTensor([0] * domain_pred_t.size(0)).to(device)
            domain_true_v = torch.LongTensor([1] * domain_pred_v.size(0)).to(device)
            domain_true_a = torch.LongTensor([2] * domain_pred_a.size(0)).to(device)

            # Stack up predictions and true labels
            domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
            domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

            return self.domain_loss_criterion(domain_pred, domain_true)

        def get_diff_loss(self):
            self.loss_diff = DiffLoss()
            shared_t = self.mutimodal_fusiom.utt_shared_t
            shared_v = self.mutimodal_fusiom.utt_shared_v
            shared_a = self.mutimodal_fusiom.utt_shared_a
            private_t = self.mutimodal_fusiom.utt_private_t
            private_v = self.mutimodal_fusiom.utt_private_v
            private_a = self.mutimodal_fusiom.utt_private_a

            # Between private and shared
            loss = self.loss_diff(private_t, shared_t)
            loss += self.loss_diff(private_v, shared_v)
            loss += self.loss_diff(private_a, shared_a)

            # Across privates
            loss += self.loss_diff(private_a, private_t)
            loss += self.loss_diff(private_a, private_v)
            loss += self.loss_diff(private_t, private_v)

            return loss

        def get_recon_loss(self, ):
            self.loss_recon = MSE()
            loss = self.loss_recon(self.mutimodal_fusiom.utt_t_recon, self.mutimodal_fusiom.utt_t_orig)
            loss += self.loss_recon(self.mutimodal_fusiom.utt_v_recon, self.mutimodal_fusiom.utt_v_orig)
            loss += self.loss_recon(self.mutimodal_fusiom.utt_a_recon, self.mutimodal_fusiom.utt_a_orig)
            loss = loss/3.0
            return loss


else:
    class PoseGenerator(nn.Module):
        def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings, z_obj=None):
            super().__init__()
            self.pre_length = args.n_pre_poses
            self.gen_length = args.n_poses - args.n_pre_poses
            self.z_obj = z_obj
            self.input_context = args.input_context

            if self.input_context == 'both':
                self.in_size = args.dim_audio + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
            elif self.input_context == 'none':
                self.in_size = pose_dim + 1
            else:
                self.in_size = 32 + pose_dim + 1  # audio or text only

            self.audio_encoder = WavEncoder(args)
            self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                               dropout=args.dropout_prob)

            self.speaker_embedding = None
            if self.z_obj:
                self.z_size = 16
                self.in_size += self.z_size
                if isinstance(self.z_obj, vocab.Vocab):
                    self.speaker_embedding = nn.Sequential(
                        nn.Embedding(z_obj.n_words, self.z_size),
                        nn.Linear(self.z_size, self.z_size)
                    )
                    self.speaker_mu = nn.Linear(self.z_size, self.z_size)
                    self.speaker_logvar = nn.Linear(self.z_size, self.z_size)
                else:
                    pass  # random noise

            self.hidden_size = args.hidden_size
            self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layers, batch_first=True,
                              bidirectional=True, dropout=args.dropout_prob)
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.hidden_size // 2, pose_dim)
            )

            self.do_flatten_parameters = False
            if torch.cuda.device_count() > 1:
                self.do_flatten_parameters = True

            self.mutimodal_fusiom = nn.Linear(args.dim_audio + args.dim_text + args.dim_video, args.size_space * 3)      # 128+32+217
            encoder_layer = nn.TransformerEncoderLayer(d_model=args.size_space * 3, nhead=8)
            self.mutimodal_fusiom_ = nn.TransformerEncoder(encoder_layer, num_layers=1)

            self.con1d_pitch = nn.Sequential(
                nn.Conv1d(1, 2, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(2),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(2, 4, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(4, 8, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),
            )

            self.con1d_erengy = nn.Sequential(
                nn.Conv1d(1, 2, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(2),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(2, 4, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(4, 8, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),
            )

            self.con1d_volume = nn.Sequential(
                nn.Conv1d(1, 2, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(2),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(2, 4, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(4, 8, 18),  # (batch, 16, (36267+2*1600-1*(15-1)-1)/5+1), (batch, 16, 7891(.4))
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),
            )

            encoder_layer_1 = nn.TransformerEncoderLayer(d_model=args.size_space * 3 + 8 * 3, nhead=2)
            self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1)

            self.fusion_1 = nn.Sequential()
            self.fusion_1.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 3 + 8 * 3,
                                                                 out_features=args.size_space * 3))
            self.fusion_1.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
            self.fusion_1.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 3,
                                                                 out_features=args.size_space * 2))  # self.config.pose_dim

            if args.use_emo and args.use_txt_emo and args.use_audio_emo:
                encoder_layer_2 = nn.TransformerEncoderLayer(d_model=args.size_space * 2 + 8 + 3, nhead=3)  # 需要整除
                self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
                self.fusion_2 = nn.Sequential()
                self.fusion_2.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 2 + 8 + 3,
                                                                     out_features=args.size_space * 2))
                self.fusion_2.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_2.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2,
                                                                     out_features=args.pose_dim))  # self.config.pose_dim

            elif args.use_emo and args.use_txt_emo:
                encoder_layer_2 = nn.TransformerEncoderLayer(d_model=args.size_space * 2 + 3, nhead=7)  # 需要整除
                self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
                self.fusion_txt_emo = nn.Sequential()
                self.fusion_txt_emo.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 2 + 3,
                                                                           out_features=args.size_space * 2))
                self.fusion_txt_emo.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_txt_emo.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2,
                                                                           out_features=args.pose_dim))  # self.config.pose_dim
            elif args.use_emo and args.use_audio_emo:
                encoder_layer_2 = nn.TransformerEncoderLayer(d_model=args.size_space * 2 + 8, nhead=2)  # 需要整除
                self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
                self.fusion_audio_emo = nn.Sequential()
                self.fusion_audio_emo.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 2 + 8,
                                                                             out_features=args.size_space * 2))
                self.fusion_audio_emo.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_audio_emo.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2,
                                                                             out_features=args.pose_dim))  # self.config.pose_dim
            else:
                self.fusion_wo_emo = nn.Sequential()
                self.fusion_wo_emo.add_module('fusion_layer_1', nn.Linear(in_features=args.size_space * 3 + 8 * 3,
                                                                          out_features=args.size_space * 2))
                self.fusion_wo_emo.add_module('fusion_layer_1_activation', nn.LeakyReLU(inplace=True))
                self.fusion_wo_emo.add_module('fusion_layer_3', nn.Linear(in_features=args.size_space * 2,
                                                                          out_features=args.pose_dim))  # self.config.pose_dim

            self.args = args

        def forward(self, pre_seq, in_text, in_audio, vid_indices=None, pitch=None, energy=None, volume=None,
                    speech_emo=None, text_emo=None):
            '''
            # print(pre_seq.shape)        # torch.Size([128, 40, 217])
            # print(in_text.shape)        # torch.Size([128, 40])
            # print(in_audio.shape)       # torch.Size([128, 21333])
            '''

            decoder_hidden = None
            if self.do_flatten_parameters:
                self.gru.flatten_parameters()

            text_feat_seq = audio_feat_seq = None
            if self.input_context != 'none':
                # audio
                audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)

                # text
                text_feat_seq, _ = self.text_encoder(in_text)  # torch.Size([128, 40, 32])
                assert (audio_feat_seq.shape[1] == text_feat_seq.shape[1])

            # z vector; speaker embedding or random noise
            if self.z_obj:
                if self.speaker_embedding:
                    assert vid_indices is not None
                    z_context = self.speaker_embedding(vid_indices)  # -> (128, 16)
                    z_mu = self.speaker_mu(z_context)  # -> (128, 16)
                    z_logvar = self.speaker_logvar(z_context)  # -> (128, 16)
                    z_context = model.embedding_net.reparameterize(z_mu, z_logvar)
                else:
                    z_mu = z_logvar = None
                    z_context = torch.randn(in_text.shape[0], self.z_size, device=in_text.device)
            else:
                z_mu = z_logvar = None
                z_context = None

            if self.input_context == 'both':
                in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)
                decoder_outputs_ = self.mutimodal_fusiom(in_data)
                decoder_outputs = self.mutimodal_fusiom_(decoder_outputs_)
                length = pre_seq.size()[1]
                energy_ = self.con1d_erengy(energy.unsqueeze(1)).permute(0, 2, 1)
                pitch_ = self.con1d_erengy(pitch.unsqueeze(1)).permute(0, 2, 1)
                volume_ = self.con1d_volume(volume.unsqueeze(1)).permute(0, 2, 1)

                # pdb.set_trace()
                # torch.Size([1, 100, 384])
                h1_ = torch.cat((decoder_outputs, energy_, pitch_, volume_), dim=2)

                h1 = self.transformer_encoder_1(h1_)

                if args.use_emo and args.use_txt_emo and args.use_audio_emo:
                    h = self.fusion_1(h1)  # torch.Size([1, 100, 256])
                    speech_emo_ = speech_emo.unsqueeze(1).repeat(1, length, 1)
                    text_emo_ = text_emo.unsqueeze(1).repeat(1, length, 1)
                    h2_ = torch.cat((h, speech_emo_, text_emo_), dim=2)
                    h2 = self.transformer_encoder_2(h2_)
                    output = self.fusion_2(h2)
                elif args.use_emo and args.use_txt_emo:
                    h = self.fusion_1(h1)  # torch.Size([1, 100, 256])
                    text_emo_ = text_emo.unsqueeze(1).repeat(1, length, 1)
                    h2_ = torch.cat((h, text_emo_), dim=2)
                    h2 = self.transformer_encoder_2(h2_)
                    output = self.fusion_txt_emo(h2)
                elif args.use_emo and args.use_audio_emo:
                    h = self.fusion_1(h1)  # torch.Size([1, 100, 256])
                    speech_emo_ = speech_emo.unsqueeze(1).repeat(1, length, 1)
                    h2_ = torch.cat((h, speech_emo_), dim=2)
                    h2 = self.transformer_encoder_2(h2_)
                    output = self.fusion_audio_emo(h2)
                else:
                    output = self.fusion_wo_emo(h1)

            elif self.input_context == 'audio':
                in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
            elif self.input_context == 'text':
                in_data = torch.cat((pre_seq, text_feat_seq), dim=2)
            elif self.input_context == 'none':
                in_data = pre_seq
            else:
                assert False

            if z_context is not None:
                repeated_z = z_context.unsqueeze(1)
                repeated_z = repeated_z.repeat(1, in_data.shape[1], 1)
                in_data = torch.cat((in_data, repeated_z), dim=2)

            # # print(in_data.shape)        # torch.Size([128, 40, 281])
            # output, decoder_hidden = self.gru(in_data, decoder_hidden)
            # # print(output.shape)       # torch.Size([128, 40, 600])
            # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs, 128 40 300
            # output = self.out(output.reshape(-1, output.shape[2]))      # 128*40, 300->217
            # decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)

            return output, z_context, z_mu, z_logvar, None, None, None



class Discriminator(nn.Module):
    def __init__(self, args, input_size, n_words=None, word_embed_size=None, word_embeddings=None):
        super().__init__()
        self.input_size = input_size

        if n_words and word_embed_size:
            self.text_encoder = TextEncoderTCN(n_words, word_embed_size, word_embeddings)
            input_size += 32
        else:
            self.text_encoder = None

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_size, num_layers=args.n_layers, bidirectional=True,
                          dropout=args.dropout_prob, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(args.n_poses, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.text_encoder:
            text_feat_seq, _ = self.text_encoder(in_text)
            poses = torch.cat((poses, text_feat_seq), dim=2)

        output, decoder_hidden = self.gru(poses, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, input_size, args=None):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = nn.GRU(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(94, 1)        # 28 -> 34, to be update!

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        # print(poses.shape)      # torch.Size([128, 34, 27])
        poses = poses.transpose(1, 2)       # torch.Size([128, 27, 34])
        feat = self.pre_conv(poses)
        # print(feat.shape)       # torch.Size([128, 8, 28])
        feat = feat.transpose(1, 2)
        # print(feat.shape)       # torch.Size([128, 28, 8])
        output, decoder_hidden = self.gru(feat, decoder_hidden)
        # print(output.shape)     # torch.Size([128, 28, 128])
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        # print(output.shape)     # torch.Size([128, 28, 64])

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        # print(output.shape)     # torch.Size([128, 28])
        # pdb.set_trace()
        output = self.out2(output)
        output = torch.sigmoid(output)
        # pdb.set_trace()
        return output


if __name__ == '__main__':
    '''
    python model/multimodal_context_net.py --config=<..your path/GENEA/genea_challenge_2022/baselines/Tri/config/multimodal_context.yml>
    '''

    # audio_encoder = WavEncoder()
    # x = torch.randn(128, 53333)
    # y = audio_encoder(x)        # 128, 166, 1024
    # print(x.shape)
    # print(y.shape)



    import sys

    [sys.path.append(i) for i in ['.', '..']]
    import os
    from config.parse_args import parse_args
    args = parse_args()
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    from utils.vocab_utils import build_vocab
    from data_loader.lmdb_data_loader import TwhDataset
    train_dataset = TwhDataset(args.train_data_path[0],
                               n_poses=args.n_poses,
                               subdivision_stride=args.subdivision_stride,
                               pose_resampling_fps=args.motion_resampling_framerate,
                               data_mean=args.data_mean, data_std=args.data_std)
    val_dataset = TwhDataset(args.val_data_path[0],
                             n_poses=args.n_poses,
                             subdivision_stride=args.subdivision_stride,
                             pose_resampling_fps=args.motion_resampling_framerate,
                             data_mean=args.data_mean, data_std=args.data_std)
    lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)
    generator = PoseGenerator(args,
                              n_words=lang_model.n_words,
                              word_embed_size=args.wordembed_dim,
                              word_embeddings=lang_model.word_embedding_weights,
                              z_obj=None,
                              pose_dim=216).to(torch.device('cuda:1'))
    pre_seq = torch.rand(8, 100, 217).to(torch.device('cuda:1'))
    in_text = torch.ones(8, 100).long().to(torch.device('cuda:1'))
    in_audio = torch.rand(8, 53333).to(torch.device('cuda:1'))
    in_pitch = torch.rand(8, 151).to(torch.device('cuda:1'))
    in_energy = torch.rand(8, 151).to(torch.device('cuda:1'))
    in_volume = torch.rand(8, 151).to(torch.device('cuda:1'))
    y, *_ = generator(pre_seq, in_text, in_audio, vid_indices=None, pitch=in_pitch, energy=in_energy, volume=in_volume)
    print(y.shape)

    print(generator.get_cmd_loss())
    print(generator.get_diff_loss())
    print(generator.get_recon_loss())
    print(generator.get_domain_loss())

    '''
    audio_encoder = WavEncoder()

    # x = torch.rand(128, 36267)
    # y = audio_encoder(x)
    # print(x.shape)      # torch.Size([128, 36267])
    # print(y.shape)      # torch.Size([128, 34, 32])

    x = torch.rand(128, 21333)
    y = audio_encoder(x)
    print(x.shape)  # torch.Size([128, 21333])
    print(y.shape)  # torch.Size([128, 40, 32])
    '''

    # net = ConvDiscriminator(27)
    # x = torch.rand(128, 34, 27)
    # y = net(x)
    # print(y.shape)      # torch.Size([128, 1])

    # net = ConvDiscriminator(216)
    # x = torch.rand(128, 40, 216)
    # y = net(x)
    # print(y.shape)  # torch.Size([128, 1])
