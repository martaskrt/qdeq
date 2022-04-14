# Quantum Deep Equilibrium (QDEQ) Models

This repository is based on the deep equilibrium transformer (DEQ-Transformer) model proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun, adapted to a quantum circuit.


### Requirements
```
Python >= 3.8
TorchQuantum (this fork: https://github.com/philipp-q/torchquantum/tree/single_qubit_hotfix)
PyTorch >=1.5.0
torchvision >= 0.4.0
CodeCarbon for CO2 monitoring (https://github.com/mlco2/codecarbon)
```

This folder contains code for training Quantum Deep Equilibrium Models (QDEQ), a framework for training two quantum machine learning (QML) models using implicit differentiation. We developed two models: the first model is paramaterized quantum circuit trained to predict a Fourier series and the second is a quantum classifier for a subset of MNIST digits. 

The main training script cant be found here:
```
train_qdeq.py
```
The models can be found in:

```
models/qdeq_model.py
```

Data loading scripts for the Fourier series amd MNIST are, respectively, in:

```
load_fourier.py
load_mnist.py
```

To train and evaluate a model, you can use one of the bash scripts in this repository (for example, `qdeq_mnist.sh`). Run `bash qdeq_mnist.sh train` to train the model. You can indiciate certain flags depending on what mode you want to train in. For instance, specify `--direct` if you want to use a direct solver instead of the implidict one. `n_layer` allows you to specify how many times to apply the model in the direct solver or warmup phase. To perform a warmup phase before implicit differentation, specify `--pretrain_steps` followed by the number of pretraining steps. 

Note that you can indicate a `--cuda` flag to run the model on a GPU; however, we had to modify some PyTorch source code for it to run because of a possible bug in the RandomSampler generator. We recommend you don't run it with CUDA; otherwise, be forewarned!


