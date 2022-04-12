#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (QDEQ)...'
    python train_qdeq.py \
        --optim Adam \
        --cuda \
        --lr 0.001 \
        --pretrain_steps 0 \
        --max_step 1935 \
        --name qdeq_mnist_direct_l1 \
        --mode direct \
        --n_layer 1 \
        --f_solver anderson \
        --b_solver broyden \
        --stop_mode rel \
        --f_thres 30 \
        --b_thres 35 \
        --jac_loss_weight 0.0 \
        --jac_loss_freq 0.0 \
        --jac_incremental 0 \
        --batch_size 256 \
	--log-interval 43 \
	--eval-interval 43 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
