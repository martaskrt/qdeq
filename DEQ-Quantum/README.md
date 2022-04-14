# Quantum Deep Equilibrium (QDEQ) Models

This repository is based on the deep equilibrium transformer (DEQ-Transformer) model proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun, adapted to a quantum circuit.


### Requirements
```
Python >= 3.8
TorchQuantum (this fork: https://github.com/philipp-q/torchquantum/tree/single_qubit_hotfix)
PyTorch >=1.5.0
torchvision >= 0.4.0
```

This folder contains code for training Quantum Deep Equilibrium Models (QDEQ), a framework for training two quantum machine learning (QML) models using implicit differentiation. We developed two models: the first model is paramaterized quantum circuit trained to predict a Fourier series and the second is a quantum classifier for a subset of MNIST digits. 

The main training script cant be found here:
'''
train_qdeq.py
'''
The models can be found in:

'''
models/qdeq_model.py
'''

Data loading scripts for the Fourier series amd MNIST are, respectively, in:

'''
load_fourier.py
load_mnist.py
'''

To train and evaluate a model, you can use one of the bash scripts. 
