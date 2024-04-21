#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=40GB
#SBATCH --job-name qdeq
#SBATCH --output=./slurm_files/%j.out
#SBATCH --error=./slurm_files/%j.error

#conda activate qdeq_2


for lr in 0.05 0.01 0.0075 0.005
do
    echo 'Run training (QDEQ)...'
    python train_qdeq.py \
        --optim Adam \
	--lr $lr \
        --cuda \
        --scheduler constant \
        --max_step 3800 \
        --name qdeq_mnist_direct \
        --mode direct \
        --n_layer 2 \
        --batch_size 256 \
	--log-interval 38 \
	--eval-interval 38
done
