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

from lib.optimizations import weight_norm, VariationalDropout
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.positional_embedding import PositionalEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits



# class FourierModel(nn.Module):
class FourierModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        # Here, consider only 1-qubit problems
        self.n_wires = 1
        # Each layer has 
        # - RY(param) - RZ(param) - RY(param) - RX(data) -
        # additionally, after the last layer, there's an extra
        # parametrized RY-RZ-RY sequence
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.encoder_gates = [tqf.rx]
        # Unfortunately does not work generic but have to hard-code

        #self.rx = tqf.rx
        self.rx = tq.RX(has_params=True, trainable=False)

        self.ry00 = tq.RY(has_params=True, trainable=True)
        self.ry01 = tq.RY(has_params=True, trainable=True)
        self.ry10 = tq.RY(has_params=True, trainable=True)
        self.ry11 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RY(has_params=True, trainable=True)
        self.rz1 = tq.RY(has_params=True, trainable=True)

        self.ry20 = tq.RY(has_params=True, trainable=True)
        self.ry21 = tq.RY(has_params=True, trainable=True)
        self.ry30 = tq.RY(has_params=True, trainable=True)
        self.ry31 = tq.RY(has_params=True, trainable=True)
        self.rz2 = tq.RY(has_params=True, trainable=True)
        self.rz3 = tq.RY(has_params=True, trainable=True)

        # Peform Pauli-Z measurement
        self.measure = tq.MeasureAll(tq.PauliZ)


    def forward(self, x):
        # Reshape from (bsz, 1, 1) to (bsz,) if necessary
        x = x.reshape((x.shape[0],)) 
        self.q_device.reset_states(1)
        # encode the classical image to quantum domain
        for k, gate in enumerate(self.encoder_gates):
            gate(self.q_device, wires=k % self.n_wires, params=x[:])
        
        #self.ry00(q_device=self.q_device, wires=0)
        y = torch.zeros_like(x, dtype=torch.cfloat)
        '''
        for i, x_ in enumerate(x):
            self.q_device.reset_states(1)
            # Fix data in encoding gate
          #  for n, p in self.named_parameters():
           #     if n == 'rx.RX_params':
            #        p.data.fill_(x_)
            for k, gate in enumerate(self.encoder_gates):
                gate(self.q_device, wires=k % self.n_wires, params=[x_])
            # Extra rotations in the beginning 
           '''
        self.ry00(q_device=self.q_device, wires=0)
        self.rz0(q_device=self.q_device, wires=0)
        self.ry01(q_device=self.q_device, wires=0)

        # First layer
        self.rx(q_device=self.q_device, wires=0)

        self.ry10(q_device=self.q_device, wires=0)
        self.rz1(q_device=self.q_device, wires=0)
        self.ry11(q_device=self.q_device, wires=0)

        # Second layer
        self.rx(q_device=self.q_device, wires=0)

        self.ry20(q_device=self.q_device, wires=0)
        self.rz2(q_device=self.q_device, wires=0)
        self.ry21(q_device=self.q_device, wires=0)

        # Third layer
        self.rx(q_device=self.q_device, wires=0)

        self.ry30(q_device=self.q_device, wires=0)
        self.rz3(q_device=self.q_device, wires=0)
        self.ry31(q_device=self.q_device, wires=0)
        '''
            y[i] = self.measure(self.q_device)
        '''
        y = self.measure(self.q_device)
         
        return y

def mse_loss(x, y):
    delta = torch.square(x - y)
    loss = torch.sum(delta) / len(x)
    return loss

loss_fn_fourier = mse_loss
class QFCModel2(nn.Module):
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
        #x = F.log_softmax(x, dim=1)
        x = x.reshape(x.shape[0], 1, -1)
        out = self.rescale(x)
        return out

class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50,
                                               wires=list(range(self.n_wires)), seed=1111)

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
        
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            self.random_layer(self.q_device)

            # some trainable gates (instantiated ahead of time)
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=1)
            self.rz0(self.q_device, wires=3)
            self.crx0(self.q_device, wires=[0, 2])

            # add some more non-parameterized gates (add on-the-fly)
            tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
                         parent_graph=self.graph)
            tqf.sx(self.q_device, wires=2, static=self.static_mode,
                   parent_graph=self.graph)
            tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
                     parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.rescale = nn.Upsample(scale_factor=8)
    def forward(self, x, injection):
        bsz = x.shape[0]
        x = x.squeeze() + injection.squeeze()
        x = x.view(bsz, 16)
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        #x = F.log_softmax(x, dim=1)
        x = x.reshape(x.shape[0], 1, -1)
        out = self.rescale(x)
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
        #out = self.conv(x)
        return out.reshape(out.shape[0], 1, -1)

class CLS(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(16,2),
                                 #nn.ReLU(inplace=True),
                                 #nn.Linear(2,2)
                                 )
    def forward(self, x):
        h = self.lin(x)
        out = F.log_softmax(h, dim=1)
        #print(out)
        return out

class QDEQCircuit(nn.Module):
    def __init__(self, dataset, mode="implicit", n_layer=None, pretrain_steps=1, device=None, f_solver=anderson, b_solver=None, stop_mode="rel", logging=None):
        super().__init__()
        self.pretrain_steps = pretrain_steps

        # self.inject_conv = nn.Conv1d(d_model, 3*d_model, kernel_size=1)
        self.input_conv = ImgFilter()
        self.device = device
        self.dataset = dataset
        self.mode = mode
        self.n_layer = n_layer
        if self.dataset == "mnist":
            self.func = QFCModel().to(device)
            # classifier:
            self.cls = CLS()
        elif self.dataset == "fourier":
            self.func = FourierModel().to(device, dtype=torch.cfloat)
            # print("Fourier model not implemented yet!"); import sys; sys.exit(0)
        self.f_solver = f_solver
        self.b_solver = b_solver if b_solver else self.f_solver
        self.hook = None
        self.stop_mode = stop_mode
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.logging = logging or print
        self.iodrop = VariationalDropout()

    def _forward(self, x, debug=False, mems=None, f_thres=30, b_thres=40, train_step=-1,
                 compute_jac_loss=True, spectral_radius_mode=False, writer=None):
        if self.dataset == "mnist":
            bsz, _, W, H = x.shape
            u1s = self.input_conv(x)
            assert u1s.shape == (bsz, 1, 16)
            func_args = [u1s]
            #z1s = torch.zeros((bsz, 1, 16), requires_grad=True)
            z1s = torch.zeros((bsz, 1, 16))
        elif self.dataset == "fourier":
            # bsz, _, qlen = x.shape
            bsz = x.shape[0]
            func_args = []
            z1s = torch.zeros(bsz, 1, 1) # bsz x 1 for 1 qubit
        jac_loss = torch.tensor(0.0).to(z1s)
        sradius = torch.zeros(bsz, 1).to(z1s)
        #deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)
        #self.pretrain_steps = 0
        #print("mode", self.mode, train_step, self.pretrain_steps, self.training)


        # warming up the weights:
        if self.mode =="direct" or train_step <= self.pretrain_steps:
            #if debug:
             #   print("I AM DEBUGGING IN DIRECT SOLVER")
            for i in range(self.n_layer):
                z1s = self.func(z1s, *func_args)
            new_z1s = z1s
        elif self.mode=="implicit":
            #print("implicit solver", train_step)
            #if debug:
             #   print("I AM DEBUGGING IN IMPLICIT SOLVER")
            # Compute the equilibrium via DEQ. When in training mode, we need to register the analytical backward
            # pass according to the Theorem 1 in the paper.
            with torch.no_grad():
                # print("z1s before entering solver", torch.norm(z1s))
                result = self.f_solver(lambda z: self.func(z, *func_args), z1s, threshold=f_thres, stop_mode=self.stop_mode)
                z1s = result['result']
                # print("z1s after exiting of solver", torch.norm(z1s))
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
        core_out = new_z1s
        with torch.no_grad():
            new_z1s_plus1 = self.func(new_z1s, *func_args)
            residual = torch.mean(torch.norm(new_z1s_plus1-new_z1s, dim=1) / (torch.norm(new_z1s, dim=1)+1e-9)).item()
        if self.dataset == "mnist":
            core_out = self.iodrop(core_out, 0.05)
        core_out= core_out.reshape(bsz, -1)
        new_mems = None
        return core_out, new_mems, jac_loss.view(-1,1), sradius.view(-1,1), residual
    
    def save_weights(self, path, name='pretrained_qdeq'):
        with open(os.path.join(path, f'{name}.pth'), 'wb') as f:
            self.logging(f"Saving weight state dict at {name}.pth")
            torch.save(self.state_dict(), f)
    def forward(self, data, target, mems, debug=False, train_step=-1, **kwargs):
        f_thres = kwargs.get('f_thres', 30)
        b_thres = kwargs.get('b_thres', 40)
        compute_jac_loss = kwargs.get('compute_jac_loss', True)
        sradius_mode = kwargs.get('spectral_radius_mode', False)
        writer = kwargs.get('writer', None)
        hidden, new_mems, jac_loss, sradius, residual = self._forward(data, mems=mems, f_thres=f_thres, b_thres=b_thres, train_step=train_step, 
                                                            compute_jac_loss=compute_jac_loss, spectral_radius_mode=sradius_mode, 
                                                            writer=writer, debug=debug)
        # get prediction 
        if self.dataset == "mnist":
            pred = self.cls(hidden)
            loss = F.nll_loss(pred, target)
            #loss = nn.CrossEntropyLoss()(pred, target)
            with torch.no_grad():
                _, indices = pred.topk(1, dim=1)
                masks = indices.eq(target.view(-1, 1).expand_as(indices))
                size = target.shape[0]
                corrects = masks.sum().item()
                acc = corrects / size

        elif self.dataset == "fourier":
            # MAKE SURE COMPLEX NUMBERS ALLOWED
            pred = hidden
            loss = loss_fn_fourier(pred, target)
            acc = loss.item()

       # if new_mems is None:
        return [loss, acc, residual, jac_loss, sradius]
       # else:
       #     return [loss, acc, residual, jac_loss, sradius] + new_mems
