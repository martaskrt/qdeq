# Quantum Deep Equilibrium (QDEQ) Models

This repository is based on the deep equilibrium transformer (DEQ-Transformer) model proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun, adapted to a quantum circuit.


### Requirements
```
Python >= 3.8
TorchQuantum from github
PyTorch pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

### Installing:
- new conda environment, with python==3.8
- then, install this pytorch version:
  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
- then, git@github.com:mit-han-lab/torchquantum.git; in requirements.txt, comment out the lines with torch  and torchvision
  in your clone, run then pip install -e .



```

This folder contains code for training Quantum Deep Equilibrium Models (QDEQ), a framework for training two quantum machine learning (QML) models using implicit differentiation. We developed two models: the first model is paramaterized quantum circuit trained to predict a Fourier series and the second is a quantum classifier for a subset of MNIST digits.

The main training script cant be found here:
```
train_qdeq_temp.py
```
The models can be found in:

```
models/qdeq_model_temp.py
```

Data loading scripts for the Fourier series amd MNIST are, respectively, in:

```
load_fourier.py
load_mnist.py
```

To train and evaluate a model, you can use one of the bash scripts in this repository (for example, `qdeq_mnist.sh`). Run `bash qdeq_mnist.sh train` to train the model. You can indiciate certain flags depending on what mode you want to train in. For instance, specify `--direct` if you want to use a direct solver instead of the implidict one. `n_layer` allows you to specify how many times to apply the model in the direct solver or warmup phase. To perform a warmup phase before implicit differentation, specify `--pretrain_steps` followed by the number of pretraining steps.

Note that you can indicate a `--cuda` flag to run the model on a GPU; however, we had to modify some PyTorch source code for it to run because of a possible bug in the RandomSampler generator. We recommend you don't run it with CUDA; otherwise, be forewarned!


one setting could be
<ur python> train_qdeq_temp.py --optim Adam --lr 0.005 --cuda --scheduler constant --max_step 18800 --pretrain_steps 0 --mode implicit --n_layer 0 --name qdeq_mnist_warmup_10c --f_solver broyden --b_solver broyden --stop_mode rel --f_thres 10 --b_thres 10 --jac_loss_weight 0.8 --jac_loss_freq 1 --jac_incremental 0 --batch_size 256 --rand_f_thres_delta 2 --log-interval 188 --eval-interval 188 --num_classes 10 --n_wires 10 --dataset mnist
