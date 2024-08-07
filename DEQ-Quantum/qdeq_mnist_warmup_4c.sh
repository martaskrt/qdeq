#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=40GB
#SBATCH --job-name qdeq
#SBATCH --output=./slurm_files/%j.out
#SBATCH --error=./slurm_files/%j.error

#conda activate qdeq_2


lr=0.05
#for lr in 0.05 0.01 0.0075 0.005 do
for nps in 1875 3750 5625
do
for nl in 1 2
do

    for jlw in  0.8 0.5 0
    do
            for jlf in 1 0.8 0.5 0
        do

echo 'Run training (QDEQ)...'
python train_qdeq_temp.py \
    --optim Adam \
    --lr $lr \
    --cuda \
--scheduler constant \
    --max_step 7500 \
    --pretrain_steps $nps \
--mode implicit \
    --n_layer $nl \
    --name qdeq_mnist_warmup_4c \
    --f_solver broyden \
    --b_solver broyden \
    --stop_mode rel \
    --f_thres 10 \
    --b_thres 10 \
    --jac_loss_weight $jlw \
    --jac_loss_freq $jlf \
    --jac_incremental 0 \
    --batch_size 256 \
    --rand_f_thres_delta 2 \
--log-interval 75 \
--eval-interval 75 \
--num_classes 4 
        done
done
done
done
