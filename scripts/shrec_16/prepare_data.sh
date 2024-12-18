#!/usr/bin/env bash

#SBATCH -J prepare_data_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out


EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"


cd $EXEDIR

python ./prepare/pre_cls_dataset.py \
--data_path "/local_datasets/Meshdataset/shrec_16/raw" \
--device cuda \
--augment_orient \
