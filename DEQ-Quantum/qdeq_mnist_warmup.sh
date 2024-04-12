#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --mem=100GB
#SBATCH --job-name qdeq
#SBATCH --output=./slurm_files/%j.out
#SBATCH --error=./slurm_files/%j.error

#conda activate qdeq_2

lr=0.0005
#for lr in 0.01 0.005 0.001 0.0005 do
for ps in 190 380 
do
    for thres in 5 10 15 
    do
        for jlw in 0 0.05 0.1 0.5 1 1.5 2 
	do
            for jlf in 0 0.4 0.8 1 
	    do
                echo 'Run training (QDEQ)...'
                python train_qdeq.py \
                    --optim Adam \
                    --lr $lr \
                    --cuda \
                    --scheduler constant \
                    --max_step 3800 \
                    --pretrain_steps $ps \
                    --n_layer 2 \
                    --name qdeq_mnist_warmup \
                    --f_solver broyden \
                    --b_solver broyden \
                    --stop_mode rel \
                    --f_thres $thres \
                    --b_thres $thres \
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
