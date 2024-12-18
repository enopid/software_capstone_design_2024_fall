#!/usr/bin/env bash

#SBATCH -J cube_train
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-0
#SBATCH -o /data/thom08/rental_seraph_csj/logs/slurm-%A.out
#SBATCH -e /data/thom08/rental_seraph_csj/errorlogs/slurm-%A.out

##sh get_data.sh

##sh prepare_data.sh

EXEDIR="/data/thom08/rental_seraph_csj/repos/Laplacian2mesh"


source activate csj_vit
echo "Activate conda env."

cd $EXEDIR

## run the training
python train_meshvit_cls.py \
--mode test \
--data_path /local_datasets/Meshdataset \
--dataset cubes \
--name cubes_vit_test \
--netvit Net_GAT_Eigen_GlobalPooling \
--num_classes 22 \
--num_inputs 250 64 16 \
--lr 3e-3 \
--num_workers 16 \
--batch_size 128 \
--scheduler_mode CosWarm \
--scheduler_T0 50 \
--scheduler_eta_min 3e-7 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
--amsgrad \
--eigen_ratio 0.5 \
--voting \

python send_message.py