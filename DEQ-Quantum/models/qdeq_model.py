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


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            assert self.n_wires==4 or self.n_wires==10
            self.random_layer = tq.RandomLayer(n_ops=50,
                                               wires=list(range(self.n_wires)), seed=1111)

            # gates with trainable parameters
            self.gate_set_length = 1
            if self.n_wires == 10:
                self.gate_set_length = 4
            self.rx_list = [tq.RX(has_params=True, trainable=True) for _ in range(self.gate_set_length)]
            self.ry_list = [tq.RY(has_params=True, trainable=True) for _ in range(self.gate_set_length)]
            self.rz_list = [tq.RZ(has_params=True, trainable=True) for _ in range(self.gate_set_length)]
            self.crx_list = [tq.CRX(has_params=True, trainable=True) for _ in range(self.gate_set_length)]

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            self.random_layer(self.q_device)

            for i in range(self.gate_set_length):
                self.rx_list[i](self.q_device, wires=0+2*i)
                self.ry_list[i](self.q_device, wires=1+2*i)
                self.rz_list[i](self.q_device, wires=3+2*i)
                self.crx_list[i](self.q_device, wires=[0+2*i, 2+2*i])

            for i in range(self.gate_set_length):
                tqf.hadamard(self.q_device, wires=3+2*i, static=self.static_mode,
                             parent_graph=self.graph)
                tqf.sx(self.q_device, wires=2+2*i, static=self.static_mode,
                       parent_graph=self.graph)
                tqf.cnot(self.q_device, wires=[3+2*i, 0+2*i], static=self.static_mode,
                         parent_graph=self.graph)

    def measure_big(self):
        states = self.q_device.get_states_1d()
        return torch.square(torch.abs(states[:,:self.num_classes]))

    def __init__(self, num_classes, n_wires=10, amplitude_encoder=True):
        super().__init__()
        self.n_wires = n_wires
        assert self.n_wires==4 or self.n_wires==10
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if not amplitude_encoder:
            if self.n_wires==4:
                self.encoder = tq.GeneralEncoder(
                    tq.encoder_op_list_name_dict['4x4_ryzxy'])
            elif self.n_wires==10:
                func_list = []
                for i in range(0, 100):
                    if (i//10)%3 == 0:
                        gate = 'ry'
                    elif (i//10)%3 == 1:
                        gate = 'rx'
                    elif (i//10)%3 == 2:
                        gate = 'rz'
                    func_list += {'input_idx': [i],   'func': gate, 'wires': [i//10]},
                self.encoder = tq.GeneralEncoder(func_list)
        elif amplitude_encoder:
            self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.num_classes = num_classes
        self.upsampling_factor = self.n_wires**2 / self.num_classes
        self.rescale = nn.Upsample(scale_factor=self.upsampling_factor)

    def forward(self, x, injection):
        bsz = x.shape[0]
        x = x.squeeze() + injection.squeeze()
        x = x.view(bsz, self.n_wires**2)
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        x = x.reshape(x.shape[0], 1, -1)
        out = self.rescale(x)

        return out

class ImgFilter(nn.Module):
    def __init__(self, n_wires=10):
        super().__init__()
        self.n_wires = n_wires

        self.conv = nn.Sequential(nn.Conv2d(1, 1, 5, stride=2),
                                  nn.BatchNorm2d(1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(1, 1, 5, stride=2),
                                  nn.BatchNorm2d(1), nn.ReLU(inplace=True))
    def forward(self, x):
        if self.n_wires == 4:
            out = F.avg_pool2d(x, 6).view(x.shape[0], self.n_wires**2)
        elif self.n_wires == 10:
            out = F.avg_pool2d(x, 5, stride=3, padding=2).view(x.shape[0], self.n_wires**2)
        return out.reshape(out.shape[0], 1, -1)

class CLS(nn.Module):
    def __init__(self, num_classes, n_wires=10):
        super().__init__()
        self.lin = nn.Sequential(nn.ReLU(inplace=True), 
                                 nn.Linear(n_wires**2, num_classes),
                                 )
    def forward(self, x):
        out = self.lin(x)
        return out

class QDEQCircuit(nn.Module):
    def __init__(self, dataset, mode="implicit", n_wires=10, n_layer=None, amplitude_encoder=True, pretrain_steps=1, device=None, f_solver=anderson, b_solver=None, stop_mode="rel", logging=None, num_classes=2):
        super().__init__()
        self.pretrain_steps = pretrain_steps
        self.n_wires = n_wires
        self.num_classes = num_classes
        self.amplitude_encoder = amplitude_encoder

        self.input_conv = ImgFilter(self.n_wires)
        self.device = device
        self.dataset = dataset
        self.mode = mode
        self.n_layer = n_layer

        self.func = QFCModel(num_classes=self.num_classes, n_wires=self.n_wires, amplitude_encoder=self.amplitude_encoder).to(device)
        self.cls = CLS(num_classes=self.num_classes, n_wires=self.n_wires)

        self.f_solver = f_solver
        self.b_solver = b_solver if b_solver else self.f_solver
        self.hook = None
        self.stop_mode = stop_mode
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.logging = logging or print
        self.iodrop = VariationalDropout()

    def _forward(self, x, debug=False, mems=None, f_thres=30, b_thres=40, train_step=-1,
                 compute_jac_loss=True, spectral_radius_mode=False, writer=None):
        bsz, _, W, H = x.shape
        u1s = self.input_conv(x)
        assert u1s.shape == (bsz, 1, self.n_wires**2)
        func_args = [u1s]
        z1s = torch.zeros((bsz, 1, self.n_wires**2))
        jac_loss = torch.tensor(0.0).to(z1s)
        sradius = torch.zeros(bsz, 1).to(z1s)

        # warming up the weights:
        if self.mode =="direct" or train_step <= self.pretrain_steps:
            for i in range(self.n_layer):
                z1s = self.func(z1s, *func_args)
            new_z1s = z1s
        elif self.mode=="implicit":
            # Compute the equilibrium via DEQ. When in training mode, we need to register the analytical backward
            # pass according to the Theorem 1 in the paper.
            with torch.no_grad():
                result = self.f_solver(lambda z: self.func(z, *func_args), z1s, threshold=f_thres, stop_mode=self.stop_mode)
                z1s = result['result']
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
        pred = self.cls(hidden)
        loss = nn.CrossEntropyLoss()(pred, target)
        with torch.no_grad():
            _, indices = pred.topk(1, dim=1)
            masks = indices.eq(target.view(-1, 1).expand_as(indices))
            size = target.shape[0]
            corrects = masks.sum().item()
            acc = corrects / size

        return [loss, acc, residual, jac_loss, sradius]
