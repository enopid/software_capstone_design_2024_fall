#!/usr/bin/env bash

#SBATCH -J prepare_data_test
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p 
#SBATCH -w aurora-g6
#SBATCH -t 1-0
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out


EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"

source activate csj_vit
echo "Activate conda env."

cd $EXEDIR

python ./prepare/pre_face_cls_dataset.py \
--data_path "/local_datasets/Meshdataset/cubes/raw" \
--device cuda \
--augment_orient 
