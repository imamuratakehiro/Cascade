#!/bin/bash

#以下の文は実行するサーバのホームにnas01home...がない時、「update-nashome-symlinks」を実行するという文だが、
#全てのサーバで既に「update-nashome-symlinks」を実行済みだし、以下の文の挙動がうまくいかないので多分いらない。
#[[ -d ~/nas01home && -d ~/nas02home && -d ~/nas03home && -d ~/mrnas01home && -d ~/mrnas02home && -d ~/mrnas03home ]] || {
#    which update-nashome-symlinks >/dev/null && update-nashome-symlinks || true
#    }

# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/nas01home/linux/anaconda3/etc/profile.d/conda.sh
conda activate ${HOME}/nas01home/codes/env
# 作成したPythonスクリプトを実行
#python3 -u model_csn_640_de5.py > log.txt
#python3 -u save_cut_wav.py --start 1 --end 1 > log.txt
#python3 -u dataset_triplet.py  > log.txt
#python3 -u dataset_csv.py > log.txt
#python3 -u train.py -l -s -e 100 -b 16 > log.txt
#python3 -u train_divide.py -s -e 100 -b 32 > log.txt
#python3 -u train_unet.py -s -e 100 -b 32 > log.txt
#python3 -u csn.py > log.txt
#python3 -u test_unet.py > log.txt
#python3 -u model_csn_640_heavy.py > log.txt

#python3 -u ./run.py -s -e 400 -b 8 -m 0.2 > ./unet5/log.txt
#python3 -u ./run.py > ./unet5/log.txt
#python3 -u -m unet5 > ./unet5/log.txt

export HYDRA_FULL_ERROR=1
#python3 -u ./train.py > ./pretrain4.log
#python3 -u ./train.py > ./logfile/pretrain/to1d640_de5/128/train_mel.log
#python3 -u ./train.py experiment=pretrain/x task_name=pretrain > ./logfile/pretrain/to1d640_c64/condition32/complex/train.log
#python3 -u ./train.py experiment=pretrain/y task_name=pretrain > ./logfile/pretrain/to1d640_de5/condition32/spec/train_no_db.log
#python3 -u ./train.py experiment=pretrain/y task_name=pretrain > ./logfile/pretrain/to1d640_de5/condition32/complex/train2.log
#python3 -u ./train.py experiment=triplet_with_unet/x task_name=triplet_with_unet > ./logfile/triplet_with_unet/mlp/unet=0_triplet=1_recog=0/mel/not_list/train.log
#python3 -u ./train.py experiment=triplet_with_unet/y task_name=triplet_with_unet > ./logfile/triplet_with_unet/transformer_bpm/unet=0_triplet=1_recog=0/spec/not_list/train.log
#python3 -u ./train.py experiment=triplet_with_tfc_tdf task_name=triplet_with_tfc_tdf > ./logfile/triplet_with_tfc_tdf/unet=0_triplet=1_recog=0/mel/train.log
#python3 -u ./train.py experiment=triplet_with_hdemucs task_name=triplet_with_hdemucs > ./logfile/triplet_with_hdemucs/unet=0_triplet=1_recog=0/mel/train.log
#python3 -u ./train.py experiment=triplet_with_bsrnn task_name=triplet_with_bsrnn > ./logfile/triplet_with_bsrnn/unet=0_triplet=1_recog=0/train_embnet.log
#python3 -u ./train.py experiment=triplet_with_subband task_name=triplet_with_subband > ./logfile/triplet_with_subband/unet=0_triplet=1_recog=0/train.log
#python3 -u ./train.py experiment=triplet_with_unet/x task_name=triplet_with_unet > ./logfile/triplet_with_unet/to1d640_de5/unet=0_triplet=1_recog=0/mel/train_31ways.log
#python3 -u ./train.py > ./logfile/triplet_with_unet/to1d640_de5_addencoder/unet=0.5_triplet=1_recog=0/spec/train.log
#python3 -u ./train.py experiment=unet/x task_name=unet > ./logfile/unet/unet/spec/not_list/train.log
#python3 -u ./train.py experiment=unet/y task_name=unet > ./logfile/unet/unet/complex/train.log
#python3 -u ./train.py experiment=nnet/x task_name=nnet > ./logfile/nnet/unet=1_triplet=0_recog=0/residuals/spec/not_list/train_continue.log
#python3 -u ./train.py experiment=nnet/x task_name=nnet
#python3 -u ./train.py experiment=nnet/z task_name=nnet > ./logfile/nnet/unet=1_triplet=1_recog=0/third/drums/spec/not_list/train.log
python3 -u ./train.py experiment=nnet/w task_name=nnet > ./logfile/nnet/unet=0_triplet=1_recog=0/second/guitar/spec/not_list/sapmlp/train_lrsch.log
#python3 -u ./train.py experiment=nnet/w task_name=nnet > ./logfile/nnet/unet=0_triplet=1_recog=0/second/guitar/spec/not_list/train_all_diff2.log
#python3 -u ./train.py experiment=nnet/x task_name=nnet > ./logfile/nnet/unet=0_triplet=1_recog=0/second/guitar/spec/not_list/train_5s_lr5e-5.log
#python3 -u ./train.py experiment=nnet/x task_name=nnet > ./logfile/nnet/unet=1_triplet=1_recog=0/third/guitar/spec/not_list/train_5s_lr1e-5_2.log
#python3 -u ./train.py experiment=nnet/y task_name=nnet > ./logfile/nnet/unet=1_triplet=1_recog=0/third/drums/spec/not_list/train_5s_lr1e-5_2.log
#python3 -u ./train.py experiment=nnet_mix/x task_name=nnet_mix > ./logfile/nnet_mix/spec/not_list/train_10s_l48.log
#python3 -u ./train.py experiment=nnet_mix/y task_name=nnet_mix > ./logfile/nnet_mix/spec+bpm/not_list/train_10s_harm_l200_sr11025.log
#python3 -u ./train.py experiment=nnet_mix/z task_name=nnet_mix > ./logfile/nnet_mix/spec+bpm/not_list/train_10s_harm_h200_sr11050.log
#python3 -u ./run.py -s -e 1000 -b 16 --model jnet_128_embnet > ./aeunet5triplet/log.txt
#python3 -u ./utils/make_csv.py > ./log.txt
#python3 -u ./train.py experiment=serial_ed/x task_name=serial_ed > ./logfile/serial_ed/unet=0_triplet=1_recog=0/transformer_bpm/mel/not_list/train_f4096_eed.log