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


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, last=False) -> None:
        super().__init__()
        if last:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2),
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

class JNetSAPMLP(nn.Module):
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
        if mel:
            in_channel = (n_mels//(2**6)+1)*encoder_out_size
        else:
            in_channel = (f_size/2/(2**6)+1)*encoder_out_size
        # Encoder
        self.encoder = UNetEncoder(encoder_in_size, encoder_out_size)
        # Decoder・Embedding Network
        self.att = SelfAttentionPooling(input_dim=in_channel)
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
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)
        #conv1_out, conv2_out = self.encoder(input)

        # cross attentionで時間方向を潰す
        B, C, F, T = conv6_out.shape
        x = conv6_out.permute(0,3,1,2).reshape(B, T, C * F)
        out_att = self.att(x) # self attention pooling
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
    
def main():
    # モデルを定義
    inst_list = ["drums", "bass", "piano", "guitar", "residuals"]
    model = JNet128Embnet(inst_list=inst_list, f_size=1024)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 513, 259),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()