#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (QDEQ)...'
    python train_qdeq.py \
        --optim Adam \
        --lr 0.02 \
        --mode direct \
        --n_layer 1 \
        --dataset fourier \
        --max_step 3000 \
        --f_solver broyden \
        --b_solver broyden \
        --stop_mode rel \
        --f_thres 6 \
        --b_thres 6 \
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
