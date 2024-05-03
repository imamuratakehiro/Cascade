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

def knn_mix():
    dict_conf = {
        "num_workers": 8,
        "n_song_test": 20,
        "margin": 0.2,
    }
    cfg = OmegaConf.create(dict_conf)
    path = {}
    path["drums"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-40-07/csv/drums/normal.csv"
    path["bass"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-42-03/csv/bass/normal.csv"
    path["piano"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-43-36/csv/piano/normal.csv"
    path["guitar"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-45-11/csv/guitar/normal.csv"
    inst_all = ["drums", "bass", "piano", "guitar"]
    label_vec = {}
    for inst in inst_all:
        label_vec[inst] = pd.read_csv(path[inst], header=None).values
    label_mix = label_vec["drums"][:, 0:2]
    vec_mix = np.concatenate([label_vec[inst][:, 2:] for inst in inst_all], axis=1)
    print(vec_mix.shape)
    acc = knn_psd(label=label_mix, vec=vec_mix, psd=False, cfg=cfg)
    tsne_not_psd(label=label_mix, vec=vec_mix, cfg=cfg, mode="Test", dir_path="/home/imamura23/nas02home/outputs/eval_nnet/mix/not_psd")
    print(f"Knn Accuracy mix : {acc * 100}")

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
        "recommend": 0,
        "dist": 100000
    }
    for recommend in songs:
        if target == recommend:
            continue
        dist = np.sum((cog[target] - cog[recommend])**2)
        if dist < dist_min["dist"]:
            dist_min["recommend"] = int(recommend)
            dist_min["dist"] = dist
    #print(f"{dist_min['target']:<5} => {dist_min['recommend']:<5} (distance = {dist_min['dist']:.3g})")
    return dist_min["recommend"], dist_min["dist"]

def recommend_from_emb():
    dict_conf = {
        "num_workers": 8,
        "n_song_test": 20,
        "margin": 0.2,
    }
    cfg = OmegaConf.create(dict_conf)
    path = {}
    """"""
    # transformer+BPM
    #path["drums"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-40-07/csv/drums/normal.csv"
    #path["bass"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-42-03/csv/bass/normal.csv"
    #path["piano"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-43-36/csv/piano/normal.csv"
    #path["guitar"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-45-11/csv/guitar/normal.csv"
    #path["residuals"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-10/10-07-33/csv/residuals/normal_Test_e=0.csv"
    #inst_all = ["drums", "bass", "piano", "guitar", "residuals"]

    # transformer+BPM, spec+bpm+hpsschroma, 5s, f2048_o512
    path["drums"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-04-20/csv/drums/normal_Test_e=0.csv"
    path["bass"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-10-09/csv/bass/normal_Test_e=0.csv"
    path["piano"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-12-45/csv/piano/normal_Test_e=0.csv"
    path["guitar"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-15-15/csv/guitar/normal_Test_e=0.csv"
    path["residuals"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-17-45/csv/residuals/normal_Test_e=0.csv"
    inst_all = ["drums", "bass", "piano", "guitar", "residuals"]

    # second conv+time average
    #path["drums"] = "/home/imamura23/nas02home/outputs/eval_nnet/second_psd/drums_normal_Test_e=0.csv"
    #path["bass"] = "/home/imamura23/nas02home/outputs/eval_nnet/second_psd/bass_normal_Test_e=0.csv"
    #path["piano"] = "/home/imamura23/nas02home/outputs/eval_nnet/second_psd/piano_normal_Test_e=0.csv"
    #path["guitar"] = "/home/imamura23/nas02home/outputs/eval_nnet/second_psd/guitar_normal_Test_e=0.csv"
    #inst_all = ["drums", "bass", "piano", "guitar"]

    # triplet_with_unet, spec, unet=1_triplet=1_recog=0
    path["drums"] = f"/nas02/homes/imamura23-1000067/outputs/eval_nnet/runs/2024-01-30/15-04-30/csv/drums/normal_Test_e=0.csv"
    path["bass"] = f"/nas02/homes/imamura23-1000067/outputs/eval_nnet/runs/2024-01-30/15-04-30/csv/bass/normal_Test_e=0.csv"
    path["piano"] = f"/nas02/homes/imamura23-1000067/outputs/eval_nnet/runs/2024-01-30/15-04-30/csv/piano/normal_Test_e=0.csv"
    path["guitar"] = f"/nas02/homes/imamura23-1000067/outputs/eval_nnet/runs/2024-01-30/15-04-30/csv/guitar/normal_Test_e=0.csv"
    path["residuals"] = f"/nas02/homes/imamura23-1000067/outputs/eval_nnet/runs/2024-01-30/15-04-30/csv/residuals/normal_Test_e=0.csv"
    inst_all = ["drums", "bass", "piano", "guitar", "residuals"]
    """"""
    label_vec = {}
    songs = {}
    cog = {}
    recommend = {}
    for inst in inst_all:
        #print(f"\n{inst}")
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
    label_vec["mix"] = np.concatenate([label_vec["drums"][:, 0:1]] + [label_vec[inst][:, 2:] for inst in inst_all], axis=1)
    songs["mix"] = np.unique(label_vec["mix"][:, 0])
    print(label_vec["mix"].shape)
    cog["mix"] = cul_center_of_grabity(label_vec["mix"], songs["mix"])
    recommend["mix"] = {}
    print("\nmix")
    for target in songs["mix"]:
        recommend["mix"][target] = {}
        r, d = cul_recommend_song(cog["mix"], target, songs["mix"])
        recommend["mix"][target]["recommend"], recommend["mix"][target]["dist"] = r, d
        print(f"{int(target):<5} => {recommend['mix'][target]['recommend']:<5}(dist={recommend['mix'][target]['dist']:.3g})", end=" ")
        print(f"(drums): {recommend['drums'][target]['recommend']}({recommend['drums'][target]['dist']:.3g})", end=" ")
        print(f"(bass): {recommend['bass'][target]['recommend']}({recommend['bass'][target]['dist']:.3g})", end=" ")
        print(f"(piano): {recommend['piano'][target]['recommend']}({recommend['piano'][target]['dist']:.3g})", end=" ")
        print(f"(guitar): {recommend['guitar'][target]['recommend']}({recommend['guitar'][target]['dist']:.3g})", end=" ")
        print(f"(residuals): {recommend['residuals'][target]['recommend']}({recommend['residuals'][target]['dist']:.3g})", end=" ")
        print()
    print(f"\nbidirectional")
    for target in songs["mix"]:
        recommended = recommend["mix"][target]["recommend"]
        if recommend["mix"][float(recommended)]["recommend"] == int(target):
            print(f"{int(target):<5} <=> {recommend['mix'][target]['recommend']:<5}(dist={recommend['mix'][target]['dist']:.3g})")
    for inst in inst_all:
        print(f"\n{inst}")
        for target in songs[inst]:
            print(f"{int(target):<5} => {recommend[inst][target]['recommend']:<5}(dist={recommend[inst][target]['dist']:.3g})")
    """label_mix = label_vec["drums"][:, 0:2]
    vec_mix = np.concatenate([label_vec[inst][:, 2:] for inst in inst_all], axis=1)
    print(vec_mix.shape)
    acc = knn_psd(label=label_mix, vec=vec_mix, psd=False, cfg=cfg)
    tsne_not_psd(label=label_mix, vec=vec_mix, cfg=cfg, mode="Test", dir_path="/home/imamura23/nas02home/outputs/eval_nnet/mix/not_psd")
    print(f"Knn Accuracy mix : {acc * 100}")"""

def main():
    #knn_mix()
    recommend_from_emb()

if "__main__" == __name__:
    main()

