import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from omegaconf import OmegaConf

from utils.func import knn_psd, tsne_not_psd

class MyError(Exception):
    pass

def cul_center_of_grabity(label_vec, songs):
    cog = {}
    for song in songs:
        #print(label_vec[inst][np.where(label_vec[inst][:, 0] == song)])
        vec_song = label_vec[np.where(label_vec[:, 0] == song)[0]][:, 2:]
        #print(vec_song.shape)
        cog[song] = np.sum(vec_song, axis=0) / len(vec_song)
    return cog

def cul_recommend_song(cog, target, songs):
    dist_min = {
        "target": target,
        "recommend": [],
        "dist": []
    }
    recommend_list = [0, 0, 0, 0, 0]
    dist_list = [1000, 1000, 1000, 1000, 1000]
    for recommend in songs:
        if target == recommend:
            continue
        dist = np.sum((cog[target] - cog[recommend])**2)
        for i, d_min in enumerate(dist_list):
            if dist < d_min:
                dist_list.insert(i, dist)
                dist_list = dist_list[:5]
                recommend_list.insert(i, int(recommend))
                recommend_list = recommend_list[:5]
                break
    dist_min["recommend"] = recommend_list
    dist_min["dist"] = dist_list
    print(f"{dist_min['target']:<5} => {recommend_list[0]:<5}, {recommend_list[1]:<5}, {recommend_list[2]:<5}, {recommend_list[3]:<5}, {recommend_list[4]:<5} (distance = {dist_list[0]:.3g}, {dist_list[1]:.3g}, {dist_list[2]:.3g}, {dist_list[3]:.3g}, {dist_list[4]:.3g})")
    return dist_min["recommend"], dist_min["dist"]

def recommend_from_emb():
    dict_conf = {
        "num_workers": 8,
        "n_song_test": 20,
        "margin": 0.2,
    }
    cfg = OmegaConf.create(dict_conf)
    path = {}
    # nnet_mix, spec
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-13/18-36-10/csv/mix/normal_Test_e=149.csv"
    # nnet_mix, spec+chroma
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-14/00-27-55/csv/mix/normal_Test_e=180.csv"
    # nnet_mix, spec+chroma+bpm
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-14/00-45-36/csv/mix/normal_Test_e=288.csv"
    # nnet_mix, spec+chroma+bpm, 10s f4096_o2048
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-14/01-06-28/csv/mix/normal_Test_e=133.csv"
    # nnet_mix, spec+hpsschroma+bpm, 10s, f4096_o2048
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-15/19-00-09/csv/mix/normal_Test_e=352.csv"
    # nnet_mix, spec+hpsschroma+bpm+pitchshift, 10s, f4096_o2048
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-21/23-06-45/csv/mix/normal_Test_e=357.csv"
    # nnet_mix, spec+hpsschroma+bpm+cls+pithshift, 10s, f4096_o2048
    #path["mix"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-24/18-33-50/csv/mix/normal_Test_e=254.csv"
    # nnet_mix, spec+hpsschroma+bpm+pitchshift+harm, 10s, f4096_o2048
    #path["mix"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-01-27/15-06-02/csv/mix/normal_Test_e=0.csv"
    # nnet_mix spec+bpm+harm, 10s, f4096_o2048
    #path["harm"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-01-27/23-51-46/csv/mix/normal_Test_e=0.csv"
    #path["perc"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-01-28/00-04-43/csv/mix/normal_Test_e=0.csv"
    # nnet_mix spec+bpm+pitchshift2+harm, 10s, f4096_o2048, sr11025
    #path["harm"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-29/21-58-05/csv/mix/normal_Test_e=188.csv"
    # nnet_mix, spec+bpm+pitchshift2, 10sf4096_o2048
    #path["harm"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-29/11-53-11/csv/mix/normal_Test_e=142.csv"
    #path["perc"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-29/18-23-14/csv/mix/normal_Test_e=260.csv"
    # nnet_mix, spec+bpmLpitchshift2, 10sf4096_2048, sr22050
    #path["low"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-01-30/23-14-53/csv/mix/normal_Test_e=0.csv"
    #path["middle"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-01-30/23-26-21/csv/mix/normal_Test_e=0.csv"
    #path["high"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-01-30/23-58-52/csv/mix/normal_Test_e=0.csv"
    # nnet_mix, spec, 10sf4096_o2048, sr22050
    #path["low"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-31/01-47-21/csv/mix/normal_Test_e=239.csv"
    #path["middle"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-31/01-48-18/csv/mix/normal_Test_e=179.csv"
    #path["high"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-31/01-48-51/csv/mix/normal_Test_e=134.csv"
    # nnet_mix, spec+bpm, 10sf2048_o512, sr11025, perc, harm_h200, harm_l200
    path["perc"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-31/18-02-57/csv/mix/normal_Test_e=130.csv"
    path["harm_high"] = "/home/imamura23/nas02home/outputs/eval_nnet_mix/runs/2024-02-01/02-42-05/csv/mix/normal_Test_e=0.csv"
    path["harm_low"] = "/home/imamura23/nas02home/outputs/nnet_mix/runs/2024-01-31/18-44-26/csv/mix/normal_Test_e=142.csv"
    #inst_all = ["mix"]
    #inst_all = ["harm"]
    #inst_all = ["harm", "perc"]
    #inst_all = ["low", "middle", "high"]
    #inst_all = ["low", "middle"]
    #inst_all = ["perc", "harm_high", "harm_low"]
    inst_all = ["harm_high", "harm_low"]
    label_vec = {}
    songs = {}
    cog = {}
    recommend = {}
    for inst in inst_all:
        print(f"\n{inst}")
        label_vec[inst] = pd.read_csv(path[inst], header=None).values
        songs[inst] = np.unique(label_vec[inst][:, 0])
        """cog[inst] = {}
        for song in songs[inst]:
            #print(label_vec[inst][np.where(label_vec[inst][:, 0] == song)])
            vec_song = label_vec[inst][np.where(label_vec[inst][:, 0] == song)[0]][:, 2:]
            #print(vec_song.shape)
            cog[inst][song] = np.sum(vec_song, axis=0) / len(vec_song)"""
        cog[inst] = cul_center_of_grabity(label_vec[inst], songs[inst])

        recommend[inst] = {}
        for target in songs[inst]:
            """dist_min = {
                "target": target,
                "recommend": 0,
                "dist": 100000
            }
            for recommend in songs[inst]:
                if target == recommend:
                    continue
                dist = np.sum((cog[inst][target] - cog[inst][recommend])**2)
                if dist < dist_min["dist"]:
                    dist_min["recommend"] = recommend
                    dist_min["dist"] = dist
            print(f"{dist_min['target']:<5} => {dist_min['recommend']:<5} (distance = {dist_min['dist']})")"""
            recommend[inst][target] = {}
            recommend[inst][target]["recommend"], recommend[inst][target]["dist"] = cul_recommend_song(cog[inst], target, songs[inst])
    # mix
    label_vec["mix"] = np.concatenate([label_vec[inst_all[0]][:, 0:1]] + [label_vec[inst][:, 2:] for inst in inst_all], axis=1)
    songs["mix"] = np.unique(label_vec["mix"][:, 0])
    print(label_vec["mix"].shape)
    cog["mix"] = cul_center_of_grabity(label_vec["mix"], songs["mix"])
    recommend["mix"] = {}
    print("\nmix")
    for target in songs["mix"]:
        recommend["mix"][target] = {}
        r, d = cul_recommend_song(cog["mix"], target, songs["mix"])
        recommend["mix"][target]["recommend"], recommend["mix"][target]["dist"] = r, d
        #print(f"{int(target):<5} => {recommend['mix'][target]['recommend']}(dist={recommend['mix'][target]['dist']})", end=" ")
        #print(f"(harm): {recommend['harm'][target]['recommend']}({recommend['harm'][target]['dist']:.3g})", end=" ")
        #print(f"(perc): {recommend['perc'][target]['recommend']}({recommend['perc'][target]['dist']:.3g})", end=" ")
        #print(f"(residuals): {recommend['residuals'][target]['recommend']}({recommend['residuals'][target]['dist']:.3g})", end=" ")
        #print()
    print(f"\nbidirectional")
    for target in songs["mix"]:
        recommended = recommend["mix"][target]["recommend"][0]
        if recommend["mix"][float(recommended)]["recommend"][0] == int(target):
            print(f"{int(target):<5} <=> {recommend['mix'][target]['recommend'][0]:<5}(dist={recommend['mix'][target]['dist'][0]:.3g})")

def main():
    #knn_mix()
    recommend_from_emb()

if "__main__" == __name__:
    main()

