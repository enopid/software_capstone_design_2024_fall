#!/usr/bin/env bash

#SBATCH -J manifold
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 2-0
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out

EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"


source activate csj_vit2
echo "Activate conda env."

cd $EXEDIR

## run the training
python train_meshvit_cls.py \
--data_path /local_datasets/Meshdataset \
--dataset Manifold40 \
--name manifold40_vit_0.6 \
--netvit Net_GAT_Eigen_GlobalPooling_face_V1 \
--num_classes 40 \
--num_inputs 250 64 16 \
--lr 3e-4 \
--num_workers 16 \
--batch_size 128 \
--scheduler_mode CosWarm \
--scheduler_T0 400 \
--scheduler_eta_min 3e-7 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
--epochs 200 \
--eigen_ratio 0.6 \
--face_input \
--continue_training \

python send_message.py