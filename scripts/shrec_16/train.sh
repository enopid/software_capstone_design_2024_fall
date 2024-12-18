#!/usr/bin/env bash

#SBATCH -J shrec
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -w aurora-g2
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out

EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"


source activate csj_vit2
echo "Activate conda env."

cd $EXEDIR

ulimit -c unlimited
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

## run the training
python train_meshvit_cls.py \
--data_path /local_datasets/Meshdataset \
--dataset shrec_16 \
--name shrec_16_vit \
--netvit Net_GAT_Eigen_GlobalPooling_face_V1 \
--device cuda \
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
--eigen_ratio 0.6 \
--face_input \

python send_message.py
