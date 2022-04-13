#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (QDEQ)...'
    python train_qdeq.py \
        --optim Adam \
        --cuda \
        --lr 0.005 \
        --pretrain_steps 0 \
        --max_step 1140 \
        --name qdeq_mnist_noll \
        --f_solver broyden \
        --b_solver broyden \
        --stop_mode rel \
        --f_thres 10 \
        --b_thres 10 \
        --jac_loss_weight 0.0 \
        --jac_loss_freq 0.0 \
        --jac_incremental 0 \
        --batch_size 256 \
	--log-interval 38 \
	--eval-interval 38 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
