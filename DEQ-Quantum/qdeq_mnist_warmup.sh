#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=40GB
#SBATCH --job-name qdeq
#SBATCH --output=./slurm_files/%j.out
#SBATCH --error=./slurm_files/%j.error

#conda activate qdeq_2


for lr in 0.05 0.01 0.0075 0.005 do
do
    for solver in anderson broyden
    do
    for jlw in 1 0.5 0
    do
            for jlf in 1 0.5 0 
        do
                echo 'Run training (QDEQ)...'
                python train_qdeq.py \
                    --optim Adam \
                    --lr $lr \
                    --cuda \
                    --scheduler cosine \
                    --max_step 3800 \
                    --pretrain_steps 380 \
                    --n_layer 2 \
                    --name qdeq_mnist_warmup \
                    --f_solver $solver \
                    --b_solver broyden \
                    --stop_mode rel \
                    --f_thres 10 \
                    --b_thres 10 \
                    --jac_loss_weight $jlw \
                    --jac_loss_freq $jlf \
                    --jac_incremental 0 \
                    --batch_size 256 \
                    --log-interval 38 \
                    --eval-interval 38 
            done
        done
    done
done
