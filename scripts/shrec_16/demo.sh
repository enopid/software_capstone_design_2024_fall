#!/usr/bin/env bash

#SBATCH -J train_shrec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/chltmd666/logs/slurm-%A.out
#SBATCH -e /data/chltmd666/errorlogs/slurm-%A.out

EXEDIR="/data/chltmd666/repos/Laplacian2mesh"


source activate Laplacian2mesh
echo "Activate conda env."

cd $EXEDIR

## run the training
python VITdataset_test.py \
--mode train \
--data_path /local_datasets/Meshdataset \
--dataset shrec_16 \
--name shrec_16_vit \
--num_classes 30 \
--num_inputs 250 64 16 \
--lr 3e-3 \
--batch_size 128 \
--scheduler_mode CosWarm \
--scheduler_T0 50 \
--scheduler_eta_min 3e-7 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
--amsgrad \
--num_workers 0 \
