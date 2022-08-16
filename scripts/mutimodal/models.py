import pdb

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import sys

[sys.path.append(i) for i in ['.', '..']]
from mutimodal.utils import ReverseLayerF


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.dim_text
        self.visual_size = config.dim_video
        self.acoustic_size = config.dim_audio


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between


        self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)



        ##########################################
        # mapping modalities to same sized space
        ##########################################

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=config.dim_text, out_features=config.size_space))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.size_space))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=config.dim_video, out_features=config.size_space))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.size_space))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=config.dim_audio, out_features=config.size_space))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.size_space))


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.size_space, out_features=config.size_space))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.size_space, out_features=config.size_space))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.size_space, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.size_space, out_features=4))



        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.size_space*6, out_features=self.config.size_space*4))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.size_space*4, out_features=self.config.size_space*3))        # self.config.pose_dim

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.size_space*6, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        

        
    def extract_features(self, sequence, rnn1, rnn2, layer_norm):
        # packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(sequence)
        else:
            packed_h1, final_h1 = rnn1(sequence)

        # padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(packed_h1)
        # packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            packed_h2, (final_h2, _) = rnn2(normed_h1)
        else:
            packed_h2, final_h2 = rnn2(normed_h1)

        # return final_h1, final_h2
        return packed_h1, packed_h2

    def alignment(self, sentences, visual, acoustic, lengths=None):
        
        # batch_size = sentences.size(1)      # seq_len, batch, dim
        #
        # # extract features from text modality
        # final_h1t, final_h2t = self.extract_features(sentences, self.trnn1, self.trnn2, self.tlayer_norm)       # torch.Size([2, 4, 32])
        # utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous()        # (2, 4, 64) -> (4, 2, 64)
        #
        # # extract features from visual modality
        # final_h1v, final_h2v = self.extract_features(visual, self.vrnn1, self.vrnn2, self.vlayer_norm)
        # utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous()       # (4, 868)
        #
        # # extract features from acoustic modality
        # final_h1a, final_h2a = self.extract_features(acoustic, self.arnn1, self.arnn2, self.alayer_norm)
        # utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous()       # (4, 16)

        # Shared-private encoders
        self.shared_private(sentences, visual, acoustic)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        # pdb.set_trace()
        h = torch.cat((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=2)

        # h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        # pdb.set_trace()
        o = self.fusion(h)

        # h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        # o = self.fusion(h)
        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, sentences, video, acoustic, lengths=None):
        o = self.alignment(sentences, video, acoustic, lengths)
        return o


if __name__ == '__main__':
    # from config import get_config
    # train_config = get_config(mode='train')

    '''
    python "<..your path/GENEA/genea_challenge_2022/baselines/Tri/scripts/mutimodal/models.py>" --config=<..your path/GENEA/genea_challenge_2022/baselines/Tri/config/multimodal_context.yml>
    '''

    import sys
    [sys.path.append(i) for i in ['.', '..']]
    from config.parse_args import parse_args

    args = parse_args()
    net = MISA(args)



    a = torch.rand(8, 32)
    b = torch.rand(3, 32)
    c = torch.rand(2, 32)
    d = torch.rand(1, 32)
    train_x = [a, b, c, d]

    seq_len = torch.tensor([s.size(0) for s in train_x])  # 获取数据真实的长度
    sentences = pad_sequence(train_x)
    print(sentences.shape)

    a = torch.rand(8, 217)
    b = torch.rand(3, 217)
    c = torch.rand(2, 217)
    d = torch.rand(1, 217)
    train_x = [a, b, c, d]

    video = pad_sequence(train_x)
    print(video.shape)

    a = torch.rand(8, 4)
    b = torch.rand(3, 4)
    c = torch.rand(2, 4)
    d = torch.rand(1, 4)
    train_x = [a, b, c, d]

    acoustic = pad_sequence(train_x)
    print(acoustic.shape)

    y = net(sentences.permute(1, 0, 2).contiguous(), video.permute(1, 0, 2).contiguous(), acoustic.permute(1, 0, 2).contiguous())
    print(y.shape)
