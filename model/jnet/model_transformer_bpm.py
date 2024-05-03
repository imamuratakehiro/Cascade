import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
#import soundfile
import math

from utils.func import standardize_torch, normalize_torch, destandardize_torch, denormalize_torch
from ..csn import ConditionalSimNet2d, ConditionalSimNet1d
from ..to1d.model_embedding import EmbeddingNet128to128, To1dEmbedding
from ..to1d.model_linear import To1D128timefreq, To1D128freqtime, To1D128, To1D640

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

"""class AddPositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, max_len: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()"""

class AddPositionalEncodingBasedBPM(nn.Module):
    def __init__(
        self, cfg, d_model: int, max_len: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.cfg = cfg
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor, bpm, inst) -> torch.Tensor:
        B, seq_len = x.size(0), x.size(1)
        pe = []
        #assert math.ceil(44100 / self.cfg.hop_length * 3) == x.shape[1], f"In transformer bpm, time length is not correct. {math.ceil(44100 / self.cfg.hop_length* 3)} == {x.shape[1]}"
        for b in range(B):
            # beat_lenは1拍の長さ、positional emcodingはbeat_lenの長さをつなげて作る。
            # TODO: instを決める時、かなり適当です。
            if inst == "mix":
                beat_len = math.ceil(60 / bpm[b, 0] * self.cfg.sr / self.cfg.hop_length)
            else:
                beat_len = math.ceil(60 / bpm[b, self.cfg.inst_all.index(inst)] * self.cfg.sr / self.cfg.hop_length)
            beat_len *= self.cfg.pe_bpm_len
            pe.append(torch.concat([self.positional_encoding_weight[:beat_len] for _ in range(math.ceil(seq_len / beat_len))], dim=0)[:seq_len])
        return x + torch.stack(pe, dim=0)
        #return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads_num, dropout=dropout_rate, batch_first=True)
        self.layer_norm_self_attention = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        #x = self.layer_norm_self_attention(self.__self_attention_block(x, mask) + x)
        x = self.layer_norm_self_attention(self.multi_head_attention(x, x, x)[0] + x)
        x = self.layer_norm_ffn(self.dropout_ffn(self.ffn(x)) + x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        max_len: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        #self.embedding = Embedding(vocab_size, d_model, pad_idx)

        self.positional_encoding = AddPositionalEncodingBasedBPM(cfg, d_model, max_len, device)
        #self.positional_encoding = AddPositionalEncodingBasedBPM(d_model, max_len, device)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, d_ff, heads_num, dropout_rate
                )
                for _ in range(N)
            ]
        )

    def forward(self, x: torch.Tensor, bpm, inst) -> torch.Tensor:
        x = self.positional_encoding(x, bpm, inst)
        for encoder_layer in self.encoder_layers:
            #x = encoder_layer(x, mask)
            x = encoder_layer(x)
        return x
    
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        """self.W = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, 1),
        )"""
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class TransformerWithBPM(nn.Module):
    def __init__(self, cfg, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        self.cfg = cfg
        encoder_in_size = len(inst_list)
        if not mono:
            encoder_in_size *= 2
        if cfg.complex_featurenet:
            encoder_in_size *= 2
        #encoder_out_size = len(inst_list) * 128
        if mel:
            in_channel = n_mels * encoder_in_size
        elif cfg.highpass and cfg.lowpass:
            in_channel = cfg.low_fq - cfg.high_fq
        elif not cfg.highpass and cfg.lowpass:
            in_channel = cfg.low_fq
        elif cfg.highpass and not cfg.lowpass:
            in_channel = int(f_size/2 + 1) - cfg.high_fq
        else:
            in_channel = int(f_size/2 + 1) * encoder_in_size
        if cfg.chroma_featurenet:
            in_channel += 12
        # Encoder
        if in_channel < 256: # inputのf_sizeが256(128の2倍)より小さい場合は256までlinearで拡張する
            self.ex = True
            self.expand = nn.Linear(in_channel, 256)
            in_channel = 256
            self.F = 256
        elif in_channel > 600: # in_channelが大きすぎる場合、メモリが足りないのでサイズを下げる
            self.ex = True
            self.expand = nn.Linear(in_channel, 512)
            in_channel = 512
            self.F = 512
        else:
            self.ex = False
        self.encoder = TransformerEncoder(
            cfg=cfg,
            max_len=5000,
            d_model=in_channel,
            d_ff=cfg.d_ff,
            heads_num=cfg.heads_num,
            N=cfg.n_encoder_layer,
            dropout_rate=0.1,
            device=device,
        )
        # Decoder・Embedding Network
        self.attpool = SelfAttentionPooling(input_dim=in_channel)
        out_size = len(inst_list) * 128
        if self.cfg.add_bpm:
            mlp_in_size = in_channel + 1
        else:
            mlp_in_size = in_channel
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_size, out_size * 2),
            nn.ReLU(),
            nn.Linear(out_size * 2, out_size)
        )
        self.sigmoid = nn.Sigmoid()
        #deviceを指定
        self.to(device)
        self.inst_list = inst_list

    def normalize_bpm(self, bpm):
        min = self.cfg.bpm_min; max = self.cfg.bpm_max
        return (bpm - min) / (max - min)

    def forward(self, input, bpm, inst):
        B = input.shape[0]
        if self.cfg.standardize_featurenet:
            input, mean, std = standardize_torch(input)
        elif self.cfg.normalize_featurenet:
            input, max, min = normalize_torch(input)

        # Encoder
        B, C, F, T = input.shape
        if self.ex:
            input = self.expand(input.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            F = self.F
        x = input.permute(0,3,2,1).reshape(B, T, C * F)
        x = self.encoder(x, bpm, inst)

        # cross attentionで時間方向を潰す
        out_att = self.attpool(x) # self attention pooling
        #print(out_lstm.shape)
        if self.cfg.add_bpm:
            # TODO:mixの時のbpm.shape = [B]にしか対応していない。instの時はまた作る。
            if inst == "mix":
                out_att = torch.concat([out_att, self.normalize_bpm(bpm)], dim=1)
            else:
                i = self.cfg.inst_all.index(inst)
                out_att = torch.concat([out_att, self.normalize_bpm(bpm[:, i: i+1])], dim=1)
        output_emb = self.mlp(out_att)
        csn1d = ConditionalSimNet1d()
        csn1d.to(output_emb.device)
        # 原点からのユークリッド距離にtanhをしたものを無音有音の確率とする
        if len(self.cfg.inst_list) == 1:
            output_probability = {inst: torch.log(torch.sqrt(torch.sum(output_emb**2, dim=1))) for inst in self.cfg.inst_list}
        else:
            output_probability = {inst: torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        #output_probability[inst] = self.sigmoid(recog_probability)[:,0]
        #print(output_probability[inst].shape)
        return output_emb, output_probability