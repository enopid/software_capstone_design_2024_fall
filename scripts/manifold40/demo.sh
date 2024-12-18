#!/usr/bin/env bash

#SBATCH -J demo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-0
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out

EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"


source activate csj_vit
echo "Activate conda env."

cd $EXEDIR

## run the training
python VITdataset_test.py \
--mode train \
--data_path /local_datasets/Meshdataset \
--dataset Manifold40 \
--name manifold40_vit_0.6 \
--num_classes 4 \
--num_inputs 512 256 64 \
--lr 3e-3 \
--batch_size 16 \
--scheduler_mode CosWarm  \
--scheduler_T0 50 \
--scheduler_eta_min 3e-7 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \

