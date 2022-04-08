import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import sys
import copy
import numpy as np
from termcolor import colored
import os
from torch.autograd import Variable as torchvar

import torchquantum as tq
import torchquantum.functional as tqf
sys.path.append('../../')

from lib.optimizations import weight_norm, VariationalDropout, VariationalHidDropout, VariationalAttnDropout
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.positional_embedding import PositionalEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits

class QFCModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.n_wires = 4
    self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
    self.measure = tq.MeasureAll(tq.PauliZ)
    
    self.encoder_gates = [tqf.rx] * 4 + [tqf.ry] * 4 + \
                         [tqf.rz] * 4 + [tqf.rx] * 4
    self.rx0 = tq.RX(has_params=True, trainable=True)
    self.ry0 = tq.RY(has_params=True, trainable=True)
    self.rz0 = tq.RZ(has_params=True, trainable=True)
    self.crx0 = tq.CRX(has_params=True, trainable=True)

    #self.rescale = nn.Linear(2,16)
    self.rescale = nn.Upsample(scale_factor=8)
    #self.rescale = nn.Linear(4,16)
  def forward(self, x, injection):
    bsz = x.shape[0]
    # down-sample the image
    ## x = x.tile((28,28))
    #print(x.shape, injection.shape)
    #x = x.reshape((bsz,1,28,28))
    
    x = x + injection
    x = x.view(bsz, 16)
    #print("new x", x.shape)
    #x = F.avg_pool2d(x, 6).view(bsz, 16)
    
    # reset qubit states
    self.q_device.reset_states(bsz)
    
    # encode the classical image to quantum domain
    for k, gate in enumerate(self.encoder_gates):
      gate(self.q_device, wires=k % self.n_wires, params=x[:, k])
    
    # add some trainable gates (need to instantiate ahead of time)
    self.rx0(self.q_device, wires=0)
    self.ry0(self.q_device, wires=1)
    self.rz0(self.q_device, wires=3)
    self.crx0(self.q_device, wires=[0, 2])
    
    # add some more non-parameterized gates (add on-the-fly)
    tqf.hadamard(self.q_device, wires=3)
    tqf.sx(self.q_device, wires=2)
    tqf.cnot(self.q_device, wires=[3, 0])
    tqf.qubitunitary(self.q_device, wires=[1, 2], params=[[1, 0, 0, 0],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 0, 1j],
                                                           [0, 0, -1j, 0]])
    
    # perform measurement to get expectations (back to classical domain)
    x = self.measure(self.q_device).reshape(bsz, 4)
    
    x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
    x = F.log_softmax(x, dim=1)
    x = x.reshape(x.shape[0], 1, -1)
    out = self.rescale(x)
    #out = out.reshape(out.shape[0], 1, -1)
    return out

class ImgFilter(nn.Module):
    def __init__(self):
        super().__init__()
        # downsampling model from MDEQ model
        
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 5, stride=2),
                                  nn.BatchNorm2d(1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(1, 1, 5, stride=2),
                                  nn.BatchNorm2d(1),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        out = F.avg_pool2d(x, 6).view(x.shape[0], 16)
        # out = self.conv2(x)
        return out.reshape(out.shape[0], 1, -1)

class CLS(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(16,2),
                                 #nn.ReLU(inplace=True),
                                 nn.Linear(2,2))
    def forward(self, x):
        return self.lin(x)
def square_loss(targets, predictions):
    loss = 0
    targets = targets.reshape(targets.shape[0], -1)
    predictions = predictions.reshape(targets.shape[0], -1)
    for t, p in zip(targets, predictions):
        loss += (t - p) ** 2
    loss = loss / len(targets)

    return loss

loss_fn = square_loss

class QDEQCircuit(nn.Module):
    def __init__(self, pretrain_steps=1, device=None, 
                 f_solver=anderson, b_solver=None, stop_mode="rel", logging=None):
        super().__init__()
        self.pretrain_steps = pretrain_steps

        #self.n_layer = n_layer
        #self.eval_n_layer = eval_n_layer
        # self.inject_conv = nn.Conv1d(d_model, 3*d_model, kernel_size=1)
        self.input_conv = ImgFilter()
        self.device = device
        self.func = QFCModel().to(device)
        self.f_solver = f_solver
        self.b_solver = b_solver if b_solver else self.f_solver
        self.hook = None
        self.stop_mode = stop_mode
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.logging = logging or print
        self.iodrop = VariationalDropout()
        # classifier:
        # self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.cls = CLS()

    def _forward(self, x, mems=None, f_thres=30, b_thres=40, train_step=-1,
                 compute_jac_loss=True, spectral_radius_mode=False, writer=None):
        # Assume dec_inp has shape (qlen x bsz)
        #print(x.shape)
        bsz, _, W, H = x.shape
        #bsz, _, qlen = x.shape
        ## KEEP THIS FOR IMAGES: 
        ## u1s = self.inject_conv(word_emb.transpose(1,2))      # bsz x 3*d_model x qlen
        #print("X.SHAPE", x.shape)
        u1s = self.input_conv(x)
        assert u1s.shape == (bsz, 1, 16)
        func_args = [u1s]
        # z1s = torch.zeros(bsz, 1, 1) # bsz x 1 for 1 qubit
        z1s = torch.zeros((bsz, 1, 16), requires_grad=True) # bsz x 1 for 1 qubit
        #z1s = torch.zeros((bsz, 28, 28), requires_grad=True) # bsz x 1 for 1 qubit
        #z1s = torch.zeros((bsz, 1, 1), requires_grad=True) # bsz x 1 for 1 qubit
        jac_loss = torch.tensor(0.0).to(z1s)
        sradius = torch.zeros(bsz, 1).to(z1s)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)
        # warming up the weights:
        if not deq_mode:
            n_layer = self.n_layer if self.training or train_step > 0 else self.eval_n_layer
            for i in range(n_layer):
                z1s = self.func(z1s, *func_args)
            new_z1s = z1s
        else:
            # Compute the equilibrium via DEQ. When in training mode, we need to register the analytical backward
            # pass according to the Theorem 1 in the paper.
            with torch.no_grad():
                #print("entering solver")
                #print(z1s)
                result = self.f_solver(lambda z: self.func(z, *func_args), z1s, threshold=f_thres, stop_mode=self.stop_mode)
                z1s = result['result']
                #print(z1s)
                #print("out of solver")
            new_z1s = z1s
            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    z1s.requires_grad_()
                    new_z1s = self.func(z1s, *func_args)
                _, sradius = power_method(new_z1s, z1s, n_iters=150)
            
            if self.training:
                z1s.requires_grad_()
                new_z1s = self.func(z1s, *func_args).reshape(bsz, 1, -1)
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1s, z1s, vecs=1)

                def backward_hook(grad):
                    if self.hook is not None:
                        
                        # To avoid infinite loop
                        self.hook.remove()
                        torch.cuda.synchronize()
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1s, z1s, y,
                                                                     retain_graph=True,\
                                                                     )[0] + grad,\
                                                                     torch.zeros_like(grad),\
                                                                     threshold=b_thres)['result']
                    return new_grad
                self.hook = new_z1s.register_hook(backward_hook)
        core_out= new_z1s.reshape(bsz, -1)
        #core_out = self.iodrop(new_z1s, 0.05).permute(2,0,1).contiguous()
        #core_out = new_z1s.permute(2,0,1).contiguous()       # qlen x bsz x d_model
        new_mems = None
        return core_out, new_mems, jac_loss.view(-1,1), sradius.view(-1,1)

    def forward(self, data, target, mems, train_step=-1, **kwargs):

        f_thres = kwargs.get('f_thres', 30)
        b_thres = kwargs.get('b_thres', 40)
        compute_jac_loss = kwargs.get('compute_jac_loss', True)
        sradius_mode = kwargs.get('spectral_radius_mode', False)
        writer = kwargs.get('writer', None)
        hidden, new_mems, jac_loss, sradius = self._forward(data, mems=mems, f_thres=f_thres, b_thres=b_thres, train_step=train_step, 
                                                            compute_jac_loss=compute_jac_loss, spectral_radius_mode=sradius_mode, 
                                                            writer=writer)
        # get prediction
        #pred_hid = hidden[-tgt_len:]
        #pred = hidden
        pred = self.cls(hidden)
        #loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1))
        
        loss = loss_fn(pred, target)

        #print(pred.argmax(dim=-1),target)
        with torch.no_grad():
            _, indices = pred.topk(1, dim=1)
            masks = indices.eq(target.view(-1, 1).expand_as(indices))
            size = target.shape[0]
            corrects = masks.sum().item()
            acc = corrects / size
            #print(pred.shape, target.shape)
            #print(torch.sum(pred.argmax(dim=1) == target), len(target))

            #acc = torch.sum(pred.argmax(dim=1) == target)/target.shape[0]

        #print("LOSS I AM HERE", loss.item())
        #loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss, acc, jac_loss, sradius]
        else:
            return [loss, acc, jac_loss, sradius] + new_mems
