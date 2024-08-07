# Quantum Deep Equilibrium Models

## Installing:
- new conda environment, with python==3.8
- then, install this pytorch version:
  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
- then, git@github.com:mit-han-lab/torchquantum.git; in requirements.txt, comment out the lines with torch  and torchvision
  in your clone, run then pip install -e .


## Ready to run?
in the folder DEQ-Quantum, run train_qdeq_temp.py, with appropriate parameters.

one setting could be
<ur python> train_qdeq_temp.py --optim Adam --lr 0.005 --cuda --scheduler constant --max_step 18800 --pretrain_steps 0 --mode implicit --n_layer 0 --name qdeq_mnist_warmup_10c --f_solver broyden --b_solver broyden --stop_mode rel --f_thres 10 --b_thres 10 --jac_loss_weight 0.8 --jac_loss_freq 1 --jac_incremental 0 --batch_size 256 --rand_f_thres_delta 2 --log-interval 188 --eval-interval 188 --num_classes 10 --n_wires 10 --dataset mnist
