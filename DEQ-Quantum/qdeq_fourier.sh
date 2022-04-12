#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (QDEQ)...'
    python train_qdeq.py \
        --optim Adam \
        --lr 0.05 \
        --dataset fourier \
        --max_step 3000 \
        --f_solver broyden \
        --b_solver broyden \
        --stop_mode rel \
        --f_thres 30 \
        --b_thres 35 \
        --jac_loss_weight 0.0 \
        --jac_loss_freq 0.0 \
        --jac_incremental 0 \
        --batch_size 150 \
	--log-interval 10 \
	--eval-interval 10 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
