#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=40GB
#SBATCH --job-name qdeq
#SBATCH --output=./slurm_files/%j.out
#SBATCH --error=./slurm_files/%j.error

#conda activate qdeq_2


#for lr in 0.05 0.01 0.0075 0.005 
#do
#for n in 1 2 5 10 
#do
lr=0.05
n=1
echo 'Run training (QDEQ)...'
python train_qdeq_temp.py \
    --optim Adam \
    --lr $lr \
    --cuda \
--scheduler constant \
--max_step 7500 \
--dataset fashion_mnist \
--mode direct \
    --n_layer $n \
    --name qdeq_mnist_direct1l_4c \
    --batch_size 256 \
--log-interval 75 \
--eval-interval 75 \
--num_classes 10
#    done
#done

