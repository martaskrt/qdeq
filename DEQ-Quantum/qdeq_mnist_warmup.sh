#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (QDEQ)...'
    python train_qdeq.py \
        --optim Adam \
        --lr 0.005 \
        --cuda \
        --max_step 1140 \
        --pretrain_steps 380 \
        --n_layer 3 \
        --name qdeq_mnist_warmup_10e3l_b10 \
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
