#!/usr/bin/env bash

#SBATCH -J cube_face_0.6
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-0
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out

EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"


source activate csj_vit2
echo "Activate conda env."

cd $EXEDIR

## run the training
python train_meshvit_cls.py \
--data_path /local_datasets/Meshdataset \
--dataset cubes \
--name cubes_vit_0.6 \
--netvit Net_GAT_Eigen_GlobalPooling_face_V1 \
--epochs 200 \
--num_classes 22 \
--num_inputs 250 64 16 \
--lr 3e-3 \
--batch_size 128 \
--scheduler_mode CosWarm \
--scheduler_T0 200 \
--scheduler_eta_min 3e-8 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
--amsgrad \
--eigen_ratio 0.6 \
--face_input \
--continue_training \

python send_message.py
