"""640次元をcsnでマスク、それぞれをDecode。Decoderは5つ"""

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

from ..csn import ConditionalSimNet2d

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, input):
        return self.conv(input)

class UNetEncoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # Encoder
        self.conv1 = Conv2d(in_channel, 16)
        self.conv2 = Conv2d(16, 32)
        self.conv3 = Conv2d(32, 64)
        self.conv4 = Conv2d(64, 128)
        self.conv5 = Conv2d(128, 256)
        self.conv6 = Conv2d(256, 512)
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

class UNetDecoder(nn.Module):
    def __init__(self, out_channel) -> None:
        super().__init__()
        self.deconv6_a = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv6_b = nn.Sequential(
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv5_a = nn.ConvTranspose2d(256+256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = nn.Sequential(
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv4_a = nn.ConvTranspose2d(128+128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = nn.Sequential(
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv3_a = nn.ConvTranspose2d(64+64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = nn.Sequential(
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2))
        self.deconv2_a = nn.ConvTranspose2d(32+32, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = nn.Sequential(
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.2))
        self.deconv1_a = nn.ConvTranspose2d(16+16, out_channel, kernel_size = (5, 5), stride=(2, 2), padding=2)
        #self.deconv1 = nn.Sequential(
        #                        nn.LeakyReLU(0.2),
        #                        nn.Sigmoid())
        #deviceを指定
        self.to(device)
    def forward(self, sep_feature, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv6_out = self.deconv6_a(sep_feature, output_size = conv5_out.size())
        deconv6_out = self.deconv6_b(deconv6_out)
        deconv5_out = self.deconv5_a(torch.cat([deconv6_out, conv5_out], 1), output_size = conv4_out.size())
        deconv5_out = self.deconv5_b(deconv5_out)
        deconv4_out = self.deconv4_a(torch.cat([deconv5_out, conv4_out], 1), output_size = conv3_out.size())
        deconv4_out = self.deconv4_b(deconv4_out)
        deconv3_out = self.deconv3_a(torch.cat([deconv4_out, conv3_out], 1), output_size = conv2_out.size())
        deconv3_out = self.deconv3_b(deconv3_out)
        deconv2_out = self.deconv2_a(torch.cat([deconv3_out, conv2_out], 1), output_size = conv1_out.size())
        deconv2_out = self.deconv2_b(deconv2_out)
        deconv1_out = self.deconv1_a(torch.cat([deconv2_out, conv1_out], 1), output_size = input.size())
        output = torch.sigmoid(deconv1_out)
        return output

class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_in_channel = 1
        if not cfg.mono:
            self.encoder_in_channel *= 2
        if cfg.complex_unet:
            self.encoder_in_channel *= 2
        # Encoder
        self.encoder = UNetEncoder(in_channel=self.encoder_in_channel)
        # Decoder
        decoder_out_channel = len(self.cfg.inst_list) * self.encoder_in_channel
        self.decoder = UNetDecoder(out_channel=decoder_out_channel)
        #deviceを指定
        self.to(device)

    def forward(self, input):
        # Encoder
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)

        # Decoder
        decoder_out = self.decoder.forward(conv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input)
        output = {inst: decoder_out[:, idx * self.encoder_in_channel: (idx + 1) * self.encoder_in_channel] * input for idx, inst in enumerate(self.cfg.inst_list)}
        return output
    
if "__main__" == __name__:
    # モデルを定義
    model = UNet()
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 513, 259),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)