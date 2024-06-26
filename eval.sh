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
#python3 -u ./eval.py > ./logfile/pretrain/to1d640_de5/not128/mel/eval_fix_inst128_norm.log
#python3 -u ./eval.py > ./logfile/triplet_with_unet/to1d640_de5/unet=0.01_triplet=1_recog=0.01/mel/not_list/eval.log
#python3 -u ./eval.py > ./logfile/unet/unet/complex/eval.log
#python3 -u ./eval.py # > ./logfile/nnet/unet=0_triplet=1_recog=0/second/residuals/spec/not_list/transformer_bpm/eval_all_no_silence_stem.log
python3 -u ./eval.py #> ./logfile/nnet/unet=0_triplet=1_recog=0/bass/spec/not_list/eval_not_psd_all.log
#python3 -u ./run.py -s -e 1000 -b 16 --model jnet_128_embnet > ./aeunet5triplet/log.txt
#python3 -u ./utils/make_csv.py > ./log.txt