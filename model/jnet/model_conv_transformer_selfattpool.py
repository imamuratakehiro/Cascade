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

from utils.func import standardize_torch, normalize_torch, destandardize_torch, denormalize_torch
from ..csn import ConditionalSimNet2d, ConditionalSimNet1d
from ..to1d.model_embedding import EmbeddingNet128to128, To1dEmbedding
from ..to1d.model_linear import To1D128timefreq, To1D128freqtime, To1D128, To1D640

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

class AddPositionalEncoding(nn.Module):
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

        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, d_ff, heads_num, dropout_rate
                )
                for _ in range(N)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            #x = encoder_layer(x, mask)
            x = encoder_layer(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, last=False) -> None:
        super().__init__()
        if last:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = (5, 1), stride=(2, 1), padding=(2, 0)),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = (5, 1), stride=(2, 1), padding=(2, 0)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
    def forward(self, input):
        return self.conv(input)

class UNetEncoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size):
        super().__init__()
        # Encoder
        self.conv1 = Conv2d(encoder_in_size, 16)
        self.conv2 = Conv2d(16, 32)
        self.conv3 = Conv2d(32, 64)
        self.conv4 = Conv2d(64, 128)
        self.conv5 = Conv2d(128, 256)
        self.conv6 = Conv2d(256, encoder_out_size, last=True)
        #deviceを指定
        self.to(device)
    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out
    
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        #self.W = nn.Linear(input_dim, 1)
        self.W = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, 1),
        )
        
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

class ConvTransformerSAPMLP(nn.Module):
    def __init__(self, cfg, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        self.cfg = cfg
        encoder_in_size = len(inst_list)
        if not mono:
            encoder_in_size *= 2
        if cfg.complex_featurenet:
            encoder_in_size *= 2
        #encoder_out_size = len(inst_list) * 128
        if len(self.cfg.inst_list) == 1:
            encoder_out_size = 512
        else:
            encoder_out_size = len(inst_list) * 128
        # Encoder
        self.conv_encoder = UNetEncoder(encoder_in_size=encoder_in_size, encoder_out_size=encoder_out_size)
        if mel:
            in_channel = (n_mels//(2**6))*encoder_out_size
        else:
            in_channel = (f_size/2/(2**6)+1)*encoder_out_size
        self.transformer_encoder = TransformerEncoder(
            max_len=10000,
            d_model=in_channel,
            d_ff=in_channel * 2,
            heads_num=cfg.heads_num,
            N=cfg.n_encoder_layer,
            dropout_rate=0.1,
            device=device,
        )
        """te_layer = nn.TransformerEncoderLayer(
            d_model=in_channel,
            nhead=cfg.heads_num,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=te_layer,
            num_layers=cfg.n_encoder_layer,
            enable_nested_tensor=True,
        )"""
        # Decoder・Embedding Network
        self.attpool = SelfAttentionPooling(input_dim=in_channel)
        out_size = len(inst_list) * 128
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, out_size * 2),
            nn.ReLU(),
            nn.Linear(out_size * 2, out_size)
        )
        self.sigmoid = nn.Sigmoid()
        #deviceを指定
        self.to(device)
        self.inst_list = inst_list

    def forward(self, input):
        B = input.shape[0]
        if self.cfg.standardize_featurenet:
            input, mean, std = standardize_torch(input)
        elif self.cfg.normalize_featurenet:
            input, max, min = normalize_torch(input)

        # Encoder
        _, _, _, _, _, x = self.conv_encoder(input)
        B, C, F, T = x.shape
        x = x.permute(0,3,2,1).reshape(B, T, C * F)
        x = self.transformer_encoder(x)

        # cross attentionで時間方向を潰す
        out_att = self.attpool(x) # self attention pooling
        #print(out_lstm.shape)
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