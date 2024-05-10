from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.nn import ModuleList as MList, ModuleDict as MDict
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
import soundfile
import numpy as np
import random
import pandas as pd
import museval
import json

from utils.func import file_exist, knn_psd, tsne_psd, tsne_psd_marker, TorchSTFT, tsne_not_psd
from ..csn import ConditionalSimNet1d
from ..tripletnet import CS_Tripletnet

class NNet(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        featurenet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cfg,
        ckpt_model_path_unet,
        ckpt_model_path_featurenet,
        ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # network
        self.unet = unet
        self.featurenet = featurenet
        # loading pretrained model
        model_checkpoint_unet = {}
        if ckpt_model_path_unet is not None:
            print("== Loading pretrained model (unet)...")
            checkpoint = torch.load(ckpt_model_path_unet)
            """for key in checkpoint["state_dict"]:
                model_checkpoint_unet[key.replace("net.","")] = checkpoint["state_dict"][key]
            self.unet.load_state_dict(model_checkpoint_unet)"""
            for key in checkpoint["state_dict"]:
                if "unet." in key:
                    model_checkpoint_unet[key.replace("unet.","")] = checkpoint["state_dict"][key]
            self.unet.load_state_dict(model_checkpoint_unet)
            print("== pretrained model was loaded!")
        model_checkpoint_featurenet = {}
        if ckpt_model_path_featurenet is not None:
            print("== Loading pretrained model (featurenet)...")
            checkpoint = torch.load(ckpt_model_path_featurenet)
            for key in checkpoint["state_dict"]:
                if "featurenet." in key:
                    model_checkpoint_featurenet[key.replace("featurenet.","")] = checkpoint["state_dict"][key]
            self.featurenet.load_state_dict(model_checkpoint_featurenet)
            print("== pretrained model was loaded!")
        # model's required grad
        if not cfg.unet_required_grad:
            for param in unet.parameters():
                param.requires_grad = False
        if not cfg.featurenet_required_grad:
            for param in featurenet.parameters():
                param.requires_grad = False
        print(self.unet)
        print(self.featurenet)

        # loss function
        self.loss_unet  = nn.L1Loss(reduction="mean")
        self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="none") #バッチ平均
        self.loss_l2      = nn.MSELoss(reduction="sum")
        self.loss_mrl     = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均
        self.loss_cross_entropy = nn.BCEWithLogitsLoss(reduction="mean")

        self.song_type = ["anchor", "positive", "negative"]
        self.sep_eval_type = ["sdr", "sir", "isr", "sar"]
        self.recorder = MDict({})
        for step in ["Train", "Valid"]:
            self.recorder[step] = MDict({})
            self.recorder[step]["loss_all"] = MeanMetric()
            self.recorder[step]["loss_unet"] = MDict({type: MeanMetric() for type in self.song_type})
            self.recorder[step]["loss_unet"]["all"] = MeanMetric()
            self.recorder[step]["loss_triplet"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
            self.recorder[step]["loss_triplet"]["all"] = MeanMetric()
            self.recorder[step]["loss_recog"] = MeanMetric()
            self.recorder[step]["recog_acc"] = MDict({inst: BinaryAccuracy() for inst in cfg.inst_list})
            self.recorder[step]["dist_p"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
            self.recorder[step]["dist_n"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
        self.recorder["Test"] = MDict({})
        self.recorder["Test"]["recog_acc"] = MDict({inst: BinaryAccuracy() for inst in cfg.inst_list})
        self.recorder_psd = {}
        self.n_sound = {}
        for step in ["Valid", "Test"]:
            self.recorder_psd[step] = {}
            self.recorder_psd[step]["label"] = {}
            self.recorder_psd[step]["vec"] = {}
            self.recorder_psd[step]["sep"] = {}
            self.n_sound[step] = {}
            for psd in ["psd", "not_psd", "psd_mine", "similarity"]:
                self.recorder_psd[step]["label"][psd] = {inst: [] for inst in cfg.inst_list}
                self.recorder_psd[step]["vec"][psd] = {inst: [] for inst in cfg.inst_list}
                self.n_sound[step][psd] = {inst: 0 for inst in cfg.inst_list}
            for s in ["sdr", "sir", "isr", "sar"]:
                self.recorder_psd[step]["sep"][s] = {inst: MeanMetric() for inst in cfg.inst_list}
        #self.test_loss = MeanMetric()

        # abx_zume_2024
        self.category_abx = ["results_melody", "results_rhythm", "results_timbre", "results_total"]
        self.human_abx = json.load(open(cfg.metadata_dir + f"zume/abx_2024/results_modified.json", 'r'))
        self.mode_abx = ["XAB", "XYC"]
        self.abx_csv = {inst: [] for inst in cfg.inst_list}
        self.result_abx = MDict({})
        for step in ["Valid", "Test"]:
            self.result_abx[step] = MDict({})
            for inst in cfg.inst_list:
                self.result_abx[step][inst] = MDict({})
                for mode in self.mode_abx:
                    self.result_abx[step][inst][mode] = MDict({})
                    for c in self.category_abx:
                        self.result_abx[step][inst][mode][c] = MDict({
                        "recog_acc": MeanMetric()
                        #"recog_acc": []
                        })
        self.stft = TorchSTFT(cfg=cfg)
        if cfg.time_stretch:
            self.time_strecher = torchaudio.transforms.TimeStretch(hop_length=cfg.hop_length, n_freq=cfg.f_size//2+1)
        self.cfg = cfg
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.recorder["Valid"]["loss_all"].reset()
        for type in self.song_type:
            self.recorder["Valid"]["loss_unet"][type].reset()
        self.recorder["Valid"]["loss_unet"]["all"].reset()
        for inst in self.cfg.inst_list:
            self.recorder["Valid"]["loss_triplet"][inst].reset()
            self.recorder["Valid"]["loss_recog"][inst].reset()
            self.recorder["Valid"]["recog_acc"][inst].reset()
            self.recorder["Valid"]["dist_p"][inst].reset()
            self.recorder["Valid"]["dist_n"][inst].reset()
        self.recorder["Valid"]["loss_triplet"]["all"].reset()
        for step in ["Valid", "Test"]:
            for inst in self.cfg.inst_list:
                for mode in self.mode_abx:
                    for c in self.category_abx:
                        self.result_abx[step][inst][mode][c]["recog_acc"].reset()
    
    def get_loss_unet(self, X, y, pred_mask):
        batch = X.shape[0]
        loss = 0
        for idx, inst in enumerate(self.cfg.inst_all): #個別音源でロスを計算
            pred = X * pred_mask[inst]
            loss += self.loss_unet(pred, y[:,idx])
        return loss / len(self.cfg.inst_list)
    
    def get_loss_unet_triposi(self, y, pred, triposi):
        # triplet positionのところのみ分離ロスを計算
        batch = y.shape[0]
        loss = 0
        for idx, c in enumerate(triposi): #個別音源でロスを計算
            if y.shape[1] == 1:
                loss += self.loss_unet(pred[self.cfg.inst_all[c.item()]][idx], y[idx])
            else:
                loss += self.loss_unet(pred[self.cfg.inst_all[c.item()]][idx], y[idx, c])
        return loss / batch

    def get_loss_triplet(self, e_a, e_p, e_n, triposi):
        #batch = triposi.shape[0]
        if len(self.cfg.inst_list) == 1:
            distp = F.pairwise_distance(e_a, e_p, 2)
            distn = F.pairwise_distance(e_a, e_n, 2)
        else:
            tnet = CS_Tripletnet(ConditionalSimNet1d().to(e_a.device))
            distp, distn = tnet(e_a, e_p, e_n, triposi)
        if self.cfg.all_diff:
            # 出力結果の小さい方をpositive, 大きい方をnegativeと定義
            small_dist = torch.where(distp < distn, distp, distn)
            large_dist = torch.where(distp >= distn, distp, distn)
            distp = small_dist
            distn = large_dist
        target = torch.FloatTensor(distp.size()).fill_(1).to(distp.device) # 1で埋める
        loss = self.loss_triplet(distn, distp, target) # トリプレットロス
        loss_all = torch.sum(loss)/len(triposi)
        #print(loss.shape, triposi.shape)
        loss_inst  = {inst: torch.sum(loss[torch.where(triposi==i)])/len(torch.where(triposi==i)[0])  if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_all)}
        dist_p_all = {inst: torch.sum(distp[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_all)}
        dist_n_all = {inst: torch.sum(distn[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_all)}
        return loss_all, loss_inst, dist_p_all, dist_n_all
    
    def get_loss_recognise1(self, emb, cases, l = 1e5):
        batch = len(cases)
        loss_emb = 0
        for b in range(batch):
            for inst in emb.keys():
                c = self.cfg.inst_all.index(inst)
                if cases[b][c] == "1":
                    # 0ベクトルとembedded vectorの距離
                    dist_0 = F.pairwise_distance(emb[inst][b], torch.zeros_like(emb[inst][b], device=cases.device), 2)
                    target = torch.ones_like(dist_0).to(cases.device) #1で埋める
                    # 0ベクトルとembedded vectorの距離がmarginよりも大きくなることを期待
                    loss_emb += self.loss_mrl(dist_0, torch.zeros_like(dist_0, device=cases.device), target)
                else:
                    loss_emb += self.loss_l2(emb[inst][b], torch.zeros_like(emb[inst][b], device=cases.device))
        return loss_emb / batch # バッチ平均をとる

    def get_loss_recognise2(self, probability, cases):
        batch = len(cases)
        loss_recognise = 0
        for b in range(batch):
            for inst in probability.keys():
                c = self.cfg.inst_all.index(inst)
                # 実際の有音無音判定と予想した有音確率でクロスエントロピーロスを計算
                loss_recognise += self.loss_cross_entropy(probability[inst][b], cases[b, c])
        return loss_recognise / batch / len(self.cfg.inst_list)
    
    def stft_complex(self, wave):
        x = self.stft.stft(wave)
        *other, C, F, T = x.shape
        return torch.view_as_real(x.reshape(-1, C, F, T)).permute(0, 1, 4, 2, 3).reshape(*other, C * 2, F, T)
    
    def transform_for_unet(self, wave):
        device = wave.device
        if self.cfg.complex_unet:
            return {"spec": self.stft_complex(wave)}
        else:
            x = self.stft.stft(wave)
            x, phase = self.stft.magphase(x)
        return {"spec": x, "phase": phase}
    
    def time_stretch(self, complex_spec, train: bool):
        if train and self.cfg.time_stretch:
            rate = random.uniform(1 - self.cfg.stretch_rate, 1 + self.cfg.stretch_rate)
            return self.time_strecher(complex_spec, rate)
        else:
            return complex_spec
    
    def transform_for_featurenet(self, train, **kwargs):
        if self.cfg.wave_featurenet:
            spec = torch.concat(list(kwargs["unet_out"].values()), dim=1)
            if self.cfg.complex_unet:
                """complex -> wave"""
                z = self.stft.spec2complex(spec)
            else:
                """spec -> wave"""
                z = spec * kwargs["phase"]
            z = self.time_stretch(z, train)
            return self.stft.istft(z)
        elif self.cfg.complex_unet and self.cfg.complex_featurenet:
            """complex -> complex"""
            z = torch.concat(list(kwargs["unet_out"].values()), dim=1)
            z = self.time_stretch(z, train)
            return z
        elif self.cfg.complex_unet and not self.cfg.complex_featurenet:
            """complex -> spec"""
            x = self.time_strech(torch.concat(list(kwargs["unet_out"].values()), dim=1))
            x, _ = self.stft.magphase(x)
            if self.cfg.mel_featurenet:
                x = self.stft.mel(x)
            if self.cfg.db_featurenet:
                x = self.stft.amp2db(x)
            return x
        elif not self.cfg.complex_unet and self.cfg.complex_featurenet:
            """spec -> comlex"""
            out = []
            for inst in self.cfg.inst_list:
                stft = kwargs["unet_out"][inst] * kwargs["phase"]
                *other, C, F, T = stft.shape
                transformed = torch.view_as_real(stft.reshape(-1, C, F, T)).permute(0, 1, 4, 2, 3).reshape(*other, C * 2, F, T)
                out.append(transformed)
            z = torch.concat(out, dim=1)
            return self.time_stretch(z, train)
        elif not self.cfg.complex_unet and not self.cfg.complex_featurenet:
            """spec -> spec"""
            x = torch.concat(list(kwargs["unet_out"].values()), dim=1)
            z = x * kwargs["phase"]
            z = self.time_stretch(z, train)
            x, _ = self.stft.magphase(z)
            if self.cfg.chroma_featurenet:
                chroma, _, _ = self.stft.hpss_chroma(x)
            if self.cfg.mel_featurenet:
                x = self.stft.mel(x)
            if self.cfg.db_featurenet:
                x = self.stft.amp2db(x)
            if self.cfg.chroma_featurenet:
                x = torch.concat([x, chroma], dim=2)
            return x

    def forward(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        (a_x_wave, a_y_wave, p_x_wave, p_y_wave, n_x_wave, n_y_wave,
        sound_a, sound_p, sound_n,
        bpm_a, bpm_p, bpm_n, triposi) = batch
        # stft
        with torch.no_grad():
            a_x = self.transform_for_unet(a_x_wave); a_y = self.transform_for_unet(a_y_wave)
            p_x = self.transform_for_unet(p_x_wave); p_y = self.transform_for_unet(p_y_wave)
            n_x = self.transform_for_unet(n_x_wave); n_y = self.transform_for_unet(n_y_wave)
        if self.cfg.pseudo == "ba_4t":
            triposi = triposi[:, 0] # basicに変換
        a_x["unet_out"] = self.unet(a_x["spec"])
        p_x["unet_out"] = self.unet(p_x["spec"])
        n_x["unet_out"] = self.unet(n_x["spec"])
        a_pred = self.transform_for_featurenet(train=False, **a_x)
        p_pred = self.transform_for_featurenet(train=True, **p_x)
        n_pred = self.transform_for_featurenet(train=True, **n_x)
        if self.cfg.bpm:
            # TODO:instの決め方テキトーです
            a_e, a_prob = self.featurenet(a_pred, bpm_a, self.cfg.inst)
            p_e, p_prob = self.featurenet(p_pred, bpm_p, self.cfg.inst)
            n_e, n_prob = self.featurenet(n_pred, bpm_n, self.cfg.inst)
        else:
            a_e, a_prob = self.featurenet(a_pred)
            p_e, p_prob = self.featurenet(p_pred)
            n_e, n_prob = self.featurenet(n_pred)
        """if self.cfg.triplet_y:
            if len(self.cfg.inst_list) == 1:
                idx = self.cfg.inst_all.index(self.cfg.inst)
                a_y["unet_out"] = a_y["spec"][:, idx].unsqueeze(dim=1)
                p_y["unet_out"] = p_y["spec"][:, idx].unsqueeze(dim=1)
                n_y["unet_out"] = n_y["spec"][:, idx].unsqueeze(dim=1)
            a_teacher = self.transform_for_featurenet(**a_y)
            p_teacher = self.transform_for_featurenet(**p_y)
            n_teacher = self.transform_for_featurenet(**n_y)
            a_e_y, _ = self.featurenet(a_teacher)
            p_e_y, _ = self.featurenet(p_teacher)
            n_e_y, _ = self.featurenet(n_teacher)"""
        # get loss
        loss_unet_a = self.get_loss_unet_triposi(a_y["spec"], a_x["unet_out"], triposi)
        loss_unet_p = self.get_loss_unet_triposi(p_y["spec"], p_x["unet_out"], triposi)
        loss_unet_n = self.get_loss_unet_triposi(n_y["spec"], n_x["unet_out"], triposi)
        loss_triplet_all, loss_triplet, dist_p, dist_n = self.get_loss_triplet(a_e, p_e, n_e, triposi)
        loss_recog = self.get_loss_recognise2(a_prob, sound_a) + self.get_loss_recognise2(p_prob, sound_p) + self.get_loss_recognise2(n_prob, sound_n)
        # record loss
        loss_all = (loss_unet_a + loss_unet_p + loss_unet_n)*self.cfg.unet_rate\
                    + loss_triplet_all*self.cfg.triplet_rate\
                    + loss_recog*self.cfg.recog_rate
        loss_unet = {
            "anchor": loss_unet_a.item(),
            "positive": loss_unet_p.item(),
            "negative": loss_unet_n.item(),
            "all": loss_unet_a.item() + loss_unet_p.item() + loss_unet_n.item()
        }
        loss_triplet["all"] = loss_triplet_all.item()
        prob = {inst: torch.concat([a_prob[inst], p_prob[inst], n_prob[inst]], dim=0) for inst in self.cfg.inst_list}
        cases = torch.concat([sound_a, sound_p, sound_n], dim=0)
        return loss_all, loss_unet, loss_triplet, dist_p, dist_n, loss_recog.item(), prob, cases
    
    def model_step(self, mode:str, batch):
        loss_all, loss_unet, loss_triplet, dist_p, dist_n, loss_recog, prob, cases = self.forward(batch)
        # update and log metrics
        self.recorder[mode]["loss_all"](loss_all)
        self.log(f"{mode}/loss_all", self.recorder[mode]["loss_all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # unet
        for type in self.song_type:
            self.recorder[mode]["loss_unet"][type](loss_unet[type])
            self.log(f"{mode}/loss_unet_{type}", self.recorder[mode]["loss_unet"][type], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self.recorder[mode]["loss_unet"]["all"](loss_unet["all"])
        self.log(f"{mode}/loss_unet_all", self.recorder[mode]["loss_unet"]["all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # triplet
        for inst in self.cfg.inst_list:
            self.recorder[mode]["loss_triplet"][inst](loss_triplet[inst])
            self.recorder[mode]["dist_p"][inst](dist_p[inst])
            self.recorder[mode]["dist_n"][inst](dist_n[inst])
            self.log(f"{mode}/loss_triplet_{inst}", self.recorder[mode]["loss_triplet"][inst], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{mode}/dist_p_{inst}",       self.recorder[mode]["dist_p"][inst],       on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{mode}/dist_n_{inst}",       self.recorder[mode]["dist_n"][inst],       on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self.recorder[mode]["loss_triplet"]["all"](loss_triplet["all"])
        self.log(f"{mode}/loss_triplet_all", self.recorder[mode]["loss_triplet"]["all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # recognize
        self.recorder[mode]["loss_recog"](loss_recog)
        self.log(f"{mode}/loss_recog", self.recorder[mode]["loss_recog"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        for idx,inst in enumerate(self.cfg.inst_list):
            self.recorder[mode]["recog_acc"][inst](prob[inst], cases[:,idx])
            self.log(f"{mode}/recog_acc_{inst}", self.recorder[mode]["recog_acc"][inst], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        # return loss or backpropagation will fail
        return loss_all
    
    def model_step_knn_tsne(self, mode:str, batch, idx, psd: str):
        ID, ver, seg, data_wave, bpm, c = batch
        with torch.no_grad():
            data = self.transform_for_unet(data_wave)
        data["unet_out"] = self.unet(data["spec"])
        pred = self.transform_for_featurenet(train=False, **data)
        if self.cfg.bpm:
            # TODO:instの決め方てきとー
            embvec, _ = self.featurenet(pred, bpm, self.cfg.inst)
        else:
            embvec, _ = self.featurenet(pred)
        if self.cfg.test_valid_norm:
            embvec = torch.nn.functional.normalize(embvec, dim=1)
        if len(self.cfg.inst_list) == 1:
            self.recorder_psd[mode]["label"][psd][self.cfg.inst].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec"][psd][self.cfg.inst].append(embvec)
        else:
            csn_valid = ConditionalSimNet1d().to(embvec.device)
            self.recorder_psd[mode]["label"][psd][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec"][psd][self.cfg.inst_list[idx]].append(csn_valid(embvec, c))
        #if self.n_sound[mode][psd][self.cfg.inst_list[idx]] < 5:
        inst = self.cfg.inst_list[idx]
        b_sound = random.sample(range(data_wave.shape[0]), 1)
        if self.cfg.complex_unet:
            z = self.stft.spec2complex(data["unet_out"][inst][b_sound])
        else:
            z = data["unet_out"][inst][0] * data["phase"][b_sound]
        sound = self.stft.istft(z)
        path = self.cfg.output_dir+f"/sound/{inst}/{mode}_e={self.current_epoch}/{psd}"
        file_exist(path)
        soundfile.write(path + f"/separate{self.n_sound[mode][psd][self.cfg.inst_list[idx]]}_{inst}.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
        soundfile.write(path + f"/mix{self.n_sound[mode][psd][self.cfg.inst_list[idx]]}_{inst}.wav", np.squeeze(data_wave[0].to("cpu").numpy()), self.cfg.sr)
        self.n_sound[mode][psd][self.cfg.inst_list[idx]] += 1
    
    def model_step_abx(self, mode: str, batch, idx):
        inst = self.cfg.inst_list[idx]
        idnt, mix_x_wave, mix_a_wave, mix_b_wave, bpm_x, bpm_a, bpm_b, c = batch
        with torch.no_grad():
            mix_x = self.transform_for_unet(mix_x_wave)
            mix_a = self.transform_for_unet(mix_a_wave)
            mix_b = self.transform_for_unet(mix_b_wave)
        mix_x["unet_out"] = self.unet(mix_x["spec"])
        mix_a["unet_out"] = self.unet(mix_a["spec"])
        mix_b["unet_out"] = self.unet(mix_b["spec"])
        pred_x = self.transform_for_featurenet(train=False, **mix_x)
        pred_a = self.transform_for_featurenet(train=False, **mix_a)
        pred_b = self.transform_for_featurenet(train=False, **mix_b)
        #print(bpm_x.shape, bpm_a.shape, bpm_b.shape)
        if self.cfg.bpm:
            # TODO:instの決め方てきとー
            emb_x, _ = self.featurenet(pred_x, bpm_x, self.cfg.inst)
            emb_a, _ = self.featurenet(pred_a, bpm_a, self.cfg.inst)
            emb_b, _ = self.featurenet(pred_b, bpm_b, self.cfg.inst)
        else:
            emb_x, _ = self.featurenet(pred_x)
            emb_a, _ = self.featurenet(pred_a)
            emb_b, _ = self.featurenet(pred_b)
        if self.cfg.test_valid_norm:
            emb_x = torch.nn.functional.normalize(emb_x, dim=1)
            emb_a = torch.nn.functional.normalize(emb_a, dim=1)
            emb_b = torch.nn.functional.normalize(emb_b, dim=1)
        dist_XA = torch.norm(emb_x - emb_a, dim=1, keepdim=True)
        dist_XB = torch.norm(emb_x - emb_b, dim=1, keepdim=True)
        idnt = torch.unsqueeze(idnt, dim=-1)
        if mode == "Test":
            self.abx_csv[inst].append(torch.concat([idnt, dist_XA, dist_XB], dim=1))

        # calculate abx result.
        def remove_nan(abx):
            nan_removed = []
            for x in abx:
                if x == "NoResult":
                    continue
                nan_removed.append(x)
            return nan_removed
        def remove_plus_minus(abx):
            plus_minus_removed = []
            for x in abx:
                if x in ["A+", "A-"]:
                    plus_minus_removed.append("A")
                elif x in ["B+" , "B-"]:
                    plus_minus_removed.append("B")
                else:
                    assert False, f"abx contains a unknown evaluation result. ({x})"
            return plus_minus_removed
        def calculate_abx(abx, eval2score, human_per_data, threshold, precise):
            def score_list(evals, eval2score):
                score = 0
                for eval in evals:
                    score += eval2score[eval]
                return score / len(evals)
            scores = {
                "results_total": -100,
                "results_melody": -100,
                "results_timbre": -100,
                "results_rhythm": -100,
            }
            for k in scores.keys():
                ab = remove_nan(abx[k])
                if len(ab) < human_per_data:
                    continue
                else:
                    score = score_list(ab, eval2score)
                    if not precise or (score > threshold or score < -threshold):
                        scores[k] = "A" if score > 0 else "B"
            return scores
        def judge_precise(abx, eval2score, threshold):
            score = 0
            for x in abx:
                score += eval2score[x]
            score /= len(abx)
            if score > threshold or score < -threshold:
                return True
            else:
                return False
        for i, idnt_per in enumerate(idnt):
            habx = self.human_abx[f"{int(idnt_per):0=5}"]
            mabx_scores = "A" if dist_XA[i] < dist_XB[i] else "B"
            if self.cfg.calc_habx:
                habx_scores = calculate_abx(habx, self.cfg.eval2score, self.cfg.human_per_data, self.cfg.threshold, self.cfg.precise)
                for c in self.category_abx:
                    if habx_scores[c] == -100:
                        continue
                    if habx_scores[c] == mabx_scores:
                        self.result_abx[mode][inst][habx["mode"]][c]["recog_acc"](1)
                    else:
                        self.result_abx[mode][inst][habx["mode"]][c]["recog_acc"](0)
            else:
                for c in self.category_abx:
                    habx_scores = remove_nan(habx[c])
                    if len(habx_scores) < self.cfg.human_per_data: continue
                    if not (not self.cfg.precise or judge_precise(habx_scores, self.cfg.eval2score, self.cfg.threshold)): continue
                    habx_scores = remove_plus_minus(habx_scores)
                    for h in habx_scores:
                        if h == mabx_scores:
                            self.result_abx[mode][inst][habx["mode"]][c]["recog_acc"](1)
                        else:
                            self.result_abx[mode][inst][habx["mode"]][c]["recog_acc"](0)
        for m_abx in self.mode_abx:
            for c in self.category_abx:
                self.log(f"{mode}/abx_recog_acc_{m_abx}_{c}", self.result_abx[mode][inst][m_abx][c]["recog_acc"], on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)


    def training_step(
        self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_all = self.model_step("Train", batch)
        return loss_all

    def print_loss(self, mode:str):
        # unet
        print("\n\n== U-Net Loss ==")
        loss_unet = {type: self.recorder[mode]["loss_unet"][type].compute() for type in self.song_type}
        print(f"{mode} average loss UNet (anchor, positive, negative)  : {loss_unet['anchor']:2f}, {loss_unet['positive']:2f}, {loss_unet['negative']:2f}")
        loss_unet_all = self.recorder[mode]["loss_unet"]["all"].compute()
        print(f"{mode} average loss UNet            (all)              : {loss_unet_all:2f}")
        # triplet
        print("\n== Triplet Loss ==")
        for inst in self.cfg.inst_list:
            loss_triplet = self.recorder[mode]["loss_triplet"][inst].compute()
            dist_p = self.recorder[mode]["dist_p"][inst].compute()
            dist_n = self.recorder[mode]["dist_n"][inst].compute()
            print(f"{mode} average loss {inst:9}(Triplet, dist_p, dist_n) : {loss_triplet:2f}, {dist_p:2f}, {dist_n:2f}")
        loss_triplet_all = self.recorder[mode]["loss_triplet"]["all"].compute()
        print(f"{mode} average loss all      (Triplet)                 : {loss_triplet_all:2f}")
        # recognize
        print("\n== Recognize ==")
        print(f"{mode} average loss Recognize     : {self.recorder[mode]['loss_recog'].compute():2f}")
        for inst in self.cfg.inst_list:
            recog_acc = self.recorder[mode]["recog_acc"][inst].compute()
            print(f"{mode} average accuracy {inst:9} : {recog_acc*100:2f} %")
        print(f"\n== {mode} average loss all : {self.recorder[mode]['loss_all'].compute():2f}\n")
    
    def output_label_vec(self, mode, epoch, inst, label, vec):
        lv = np.concatenate([label, vec], axis=1)
        dirpath = self.cfg.output_dir + f"/csv/{inst}"
        file_exist(dirpath)
        pd.DataFrame(lv, columns=["label", "other"] + [i for i in range(128)]).to_csv(dirpath + f"/normal_{mode}_e={epoch}.csv", header=False, index=False)
    
    def knn_tsne(self, mode:str, psd: str):
        print(f"\n== {psd} ==")
        acc_all = 0
        for inst in self.cfg.inst_list:
            label = torch.concat(self.recorder_psd[mode]["label"][psd][inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.recorder_psd[mode]["vec"][psd][inst], dim=0).to("cpu").numpy()
            acc = knn_psd(label, vec, self.cfg, psd=False if psd == "not_psd" else True) # knn
            self.log(f"{mode}/knn_{psd}_{inst}",acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            if psd == "psd_mine":
                tsne_psd_marker(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            elif psd == "psd":
                tsne_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            elif psd == "not_psd":
                self.output_label_vec(mode=mode, epoch=self.current_epoch, inst=inst, label=label, vec=vec)
                tsne_not_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            print(f"{mode} knn accuracy {inst:<10} {psd:<8} : {acc*100}%")
            acc_all += acc
        self.log(f"{mode}/knn_{psd}_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        print(f"\n{mode} knn accuracy average {psd:<8}   : {acc_all/len(self.cfg.inst_list)*100}%")
        self.recorder_psd[mode]["label"][psd] = {inst:[] for inst in self.cfg.inst_list}; self.recorder_psd[mode]["vec"][psd] = {inst:[] for inst in self.cfg.inst_list}
    
    def similarity(self, mode: str):
        print(f"\n== similarity ==")
        for inst in self.cfg.inst_list:
            track = torch.concat(self.recorder_psd[mode]["label"]["similarity"][inst], dim=0)[0].to("cpu").numpy() # [ID, ver]のIDだけ
            vec   = torch.concat(self.recorder_psd[mode]["vec"]["similarity"][inst], dim=0).to("cpu").numpy()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.print_loss("Train")

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx == 0:
            loss_all = self.model_step("Valid", batch)
        elif dataloader_idx > 0 and dataloader_idx < n_inst + 1:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx-1, psd="psd")
        elif dataloader_idx >= n_inst + 1 and dataloader_idx < 2*n_inst + 1:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx - 1 - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst + 1 and dataloader_idx < 3*n_inst + 1:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx - 1 - 2*n_inst, psd="psd_mine")
        elif dataloader_idx >= 3*n_inst + 1 and dataloader_idx < 4*n_inst + 1:
            self.model_step_abx("Valid", batch, dataloader_idx - 1 - 3*n_inst)
    
    def calc_abx_scores(self, mode):
        #print(self.result_abx)
        for inst in self.cfg.inst_list:
            print(f"\n{inst}")
            for m_abx in self.mode_abx:
                print(f"  {m_abx}")
                for c in self.category_abx:
                    print(f"    {c}: {self.result_abx[mode][inst][m_abx][c]['recog_acc'].compute()*100:.2f} %")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.calc_abx_scores(mode="Valid")

        self.print_loss("Valid")
        self.knn_tsne("Valid", psd="psd")
        self.knn_tsne("Valid", psd="not_psd")
        self.knn_tsne("Valid", psd="psd_mine")
        for psd in ["psd", "not_psd", "psd_mine"]:
            self.n_sound["Valid"][psd] = {inst: 0 for inst in self.cfg.inst_list}

    def evaluate_separated(self, reference, estimate):
        # assume mix as estimates
        B_r, S_r, T_r = reference.shape
        B_e, S_e, T_e = estimate.shape
        reference = torch.reshape(reference, (B_r, T_r, S_r))
        estimate  = torch.reshape(estimate, (B_e, T_e, S_e))
        if T_r > T_e:
            reference = reference[:, :T_e]
        scores = {}
        scores = {"sdr":0, "isr":0, "sir":0, "sar":0}
        # Evaluate using museval
        score = museval.evaluate(references=reference.to("cpu"), estimates=estimate.to("cpu"))
        print(len(score))
        for i,key in enumerate(list(scores.keys())):
            scores[key] = np.mean(score[i])
        return scores

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx == 0:
            '''cases_x, cases_y, cases = batch
            with torch.no_grad():
                data = self.transform_for_unet(cases_x)
            data["unet_out"] = self.unet(data["spec"])
            pred = self.transform_for_featurenet(**data)
            #if self.cfg.bpm:
                # TODO:instの決め方てきとー
            #    embvec, _ = self.featurenet(pred, bpm, self.cfg.inst)
            #else:
            embvec, _ = self.featurenet(pred)
            if self.cfg.test_valid_norm:
                embvec = torch.nn.functional.normalize(embvec, dim=1)
            """if self.cfg.complex:
                cases_x = self.stft.transform(cases_x)
                phase = None
            else:
                cases_x, phase = self.stft.transform(cases_x)
            c_e, c_prob, c_pred = self.net(cases_x)"""
            idx_inst = self.cfg.inst_all.index(self.cfg.inst)
            if not self.cfg.mel_unet:
                for idx,c in enumerate(cases):
                    if c[idx_inst] == 1:
                        #    and self.n_sound < 5):
                        #sound = self.stft.detransform(cases_x[idx], phase[idx], param[0,idx], param[1,idx])
                        #if self.cfg.complex:
                        #    sound = self.stft.detransform(cases_x[idx])
                        #else:
                        #    sound = self.stft.detransform(cases_x[idx], phase[idx])
                        #sound = self.stft.detransform(cases_x[idx], phase[idx])
                        #path = self.cfg.output_dir+f"/sound/mix"
                        #file_exist(path)
                        #soundfile.write(path + f"/separate{self.n_sound}_mix.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                        for j,inst in enumerate(self.cfg.inst_list):
                            #sound = self.stft.detransform(cases_x[idx]*c_mask[inst][idx], phase[idx], param[0,idx], param[1,idx])
                            if self.cfg.complex_unet:
                                z = self.stft.spec2complex(data["unet_out"][inst][idx])
                            else:
                                z = data["unet_out"][inst][idx] * data["phase"][idx]
                            sound = self.stft.istft(z)
                            """if self.cfg.complex:
                                sound = self.stft.detransform(c_pred[inst][idx])
                            else:
                                sound = self.stft.detransform(c_pred[inst][idx], phase[idx])"""
                            #path = self.cfg.output_dir+f"/sound/{inst}"
                            #file_exist(path)
                            #soundfile.write(path + f"/separate{self.n_sound}_{inst}.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                            try:
                                scores = self.evaluate_separated(cases_y[idx, idx_inst: idx_inst + 1], torch.unsqueeze(sound, dim=1))
                                for s in self.sep_eval_type:
                                    self.recorder_psd["Test"]["sep"][s][inst](scores[s])
                            except:
                                pass
                        #self.n_sound += 1
            #for idx,inst in enumerate(self.cfg.inst_list):
            #    self.recorder["Test"]["recog_acc"][inst](c_prob[inst], cases[:,idx])'''
            pass
        elif dataloader_idx > 0 and dataloader_idx < n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx-1, psd="psd")
        elif dataloader_idx >= n_inst + 1 and dataloader_idx < 2*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst + 1 and dataloader_idx < 3*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - 2*n_inst, psd="psd_mine")
        elif dataloader_idx >= 3*n_inst + 1 and dataloader_idx < 4*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - 3*n_inst, psd="similarity")
        elif dataloader_idx >= 4*n_inst + 1 and dataloader_idx < 5*n_inst + 1:
            self.model_step_abx("Test", batch, dataloader_idx - 1 - 4*n_inst)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        print()
        for inst in self.cfg.inst_list:
            for s in self.sep_eval_type:
                print(f"{s} {inst:9}: {self.recorder_psd['Test']['sep'][s][inst].compute()}")
        # abx
        self.calc_abx_scores(mode="Test")
        for inst in self.cfg.inst_list:
            csv = torch.concat(self.abx_csv[inst], dim = 0).to("cpu").numpy()
            file_exist(self.cfg.output_dir + f"/csv/abx2024/{inst}")
            pd.DataFrame(csv, columns=["identifier", "dist_XA", "dist_XB"]).to_csv(self.cfg.output_dir + f"/csv/abx2024/{inst}/result.csv", index=False)
        self.knn_tsne("Test", psd="psd")
        self.knn_tsne("Test", psd="not_psd")
        self.knn_tsne("Test", psd="psd_mine")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "Valid/loss_all",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    pass