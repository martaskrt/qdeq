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

import tequila as tq
from tequila.objective import Variable as tqvar
from tequila.ml.interface_torch import TorchLayer
from tequila import QTensor

sys.path.append('../../')

from lib.optimizations import weight_norm, VariationalDropout, VariationalHidDropout, VariationalAttnDropout
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.positional_embedding import PositionalEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits



degree = 1  # degree of the target function
scaling = 1  # scaling of the data
coeffs = [0.15 + 0.15j]*degree  # coefficients of non-zero frequencies
coeff0 = 0.1  # coefficient of zero frequency

def S(x):
    """Data-encoding circuit block."""
    # qml.RX(scaling * x, wires=0)
    return tq.circuit.gates.Rx(angle=scaling*x, target=0)

def W(theta):
    """Trainable circuit block."""
    # qml.Rot = Rz(t0)Ry(t1)Rz(t2)
    # qml.Rot(theta[0], theta[1], theta[2], wires=0)
    ''' have to do the following if wanna do simulate '''
    if not isinstance(theta[0], tqvar):
        theta = [float(t) for t in theta]
    W  = tq.circuit.circuit.QCircuit()
    W += tq.circuit.gates.Rz(angle=theta[0], target=0)
    W += tq.circuit.gates.Ry(angle=theta[1], target=0)
    W += tq.circuit.gates.Rz(angle=theta[2], target=0)
    return W


def serial_quantum_model(weights, x):
    ''' quantum circuit '''
    U = tq.circuit.circuit.QCircuit()
    for theta in weights[:-1]:
        U += W(theta)
        U += S(x)
    # (L+1)'th unitary
    U += W(weights[-1])
    return U

def serial_quantum_eval(weights, x):
    ''' expectation value of quantum circuit '''
    U = serial_quantum_model(weights, x)
    H = tq.hamiltonian.QubitHamiltonian.from_string("Z(0)")
    E = tq.ExpectationValue(H=H, U=U)
    return E

class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()
        r = 2
        weights = 2 * np.pi * np.random.random(size=(r+1, 3)) # some random initial weights
        weights = torch.tensor(weights, requires_grad=True)
        old_weights = weights.cpu().detach().numpy()
        old_weights_flat = old_weights.flatten()
        weights = [[ tqvar(name=str(str(i)+str(j))) for j in range(weights.shape[1]) ] for i in range(weights.shape[0])]
        x_var = tqvar(name='x')
        #y_var = tqvar(name='y')
        
        self.qmodel = serial_quantum_eval(weights, x_var)
        variables = self.qmodel.extract_variables()
        weight_vars = copy.deepcopy(variables)
        weight_vars.remove('x')
        #variables.remove('x')
        init_vars = {v: torch.tensor([[[old_weights_flat[i]]]]) for i, v in enumerate(weight_vars)}
        init_vars['x'] = torch.tensor([[[0.]]], requires_grad=False)
        compile_args = {'backend': 'cirq', 'initial_values': init_vars}
        self.circuit = TorchLayer(self.qmodel,compile_args) 
        print(self.circuit)
        #self.circuit = TorchLayer(self.qmodel,compile_args, input_vars=[x_var]) 

    def forward(self, x):
        # If errors: might have to use or might be able to use a QTensor in the test loop
        # out = QTensor(shape=[len(x)])
        out = torch.zeros(size=(len(x),))
        for i, x_ in enumerate(x):
            for n, p in self.circuit.named_parameters():
                if n == 'x':
                    p.data = torch.tensor([[[x_]]], requires_grad=False)
            out[i] = torch.tensor([self.circuit()], requires_grad=True)
            #out[i] = self.circuit(torch.tensor([x_], requires_grad=True))
        out = out.reshape((1,1,1))
        print('returning....', out)
        
        return out

class QModel2(nn.Module):
    def __init__(self):
        super(QModel2, self).__init__()

        self.lin = nn.Linear(1,1)
    def forward(self,x):
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
        self.device = device
        self.func = QModel().to(device)
        self.f_solver = f_solver
        self.b_solver = b_solver if b_solver else self.f_solver
        self.hook = None
        self.stop_mode = stop_mode
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.logging = logging or print
        self.iodrop = VariationalDropout()
        # classifier:
        # self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)


    def _forward(self, x, mems=None, f_thres=30, b_thres=40, train_step=-1,
                 compute_jac_loss=True, spectral_radius_mode=False, writer=None):
        # Assume dec_inp has shape (qlen x bsz)
        bsz, _, qlen = x.shape
        ## KEEP THIS FOR IMAGES: 
        ## u1s = self.inject_conv(word_emb.transpose(1,2))      # bsz x 3*d_model x qlen


        # z1s = torch.zeros(bsz, 1, 1) # bsz x 1 for 1 qubit
        z1s = torch.zeros((bsz, 1, 1), requires_grad=True) # bsz x 1 for 1 qubit
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
                print("entering solver")
                print('z1s', z1s)
                print('func', self.func)
                result = self.f_solver(lambda z: self.func(z), z1s, threshold=f_thres, stop_mode=self.stop_mode)
                z1s = result['result']
                print(z1s)
                print("out of solver")
            new_z1s = z1s
            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    z1s.requires_grad_()
                    new_z1s = self.func(z1s, *func_args)
                _, sradius = power_method(new_z1s, z1s, n_iters=150)
            
            if self.training:
                z1s.requires_grad_()
                # new_z1s = self.func(z1s).reshape(bsz, -1)
                new_z1s = self.func(z1s).reshape(bsz, 1, -1)
                #new_z1s = self.func(z1s, *func_args)
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1s, z1s, vecs=1)

                def backward_hook(grad):
                    if self.hook is not None:
                        
                        # To avoid infinite loop
                        self.hook.remove()
                        torch.cuda.synchronize()
                    # fy = lambda y: autograd.grad(new_z1s, z1s, y,
                    #                               retain_graph=True,\
                    #                               allow_unused=True)[0] 

                    # fy3 = lambda y: autograd.grad(new_z1s, z1s, y,
                    #                               retain_graph=True,\
                    #                               )[0] 
                    # print(fy3(3))
                    # print(fy(3))
                    my_grad = autograd.functional.jacobian(self.func, z1s)
                    print("jac", my_grad)
                    print(grad, grad.shape)
                    grad = grad.reshape((1,1,1))
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1s, z1s, y,
                                                                     retain_graph=True,\
                                                                     allow_unused=True,\
                                                                     )[0] + grad,\
                                                                     torch.zeros_like(grad),\
                                                                     threshold=b_thres)['result']
                    return new_grad
                self.hook = new_z1s.register_hook(backward_hook)
        core_out= new_z1s.reshape(bsz, -1)
        # core_out = self.iodrop(new_z1s, 0.05).permute(2,0,1).contiguous()
        #core_out = new_z1s.permute(2,0,1).contiguous()       # qlen x bsz x d_model
        #new_mems = self._update_mems(new_z1s, us, z0, mlen, qlen)
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
        pred = hidden
        #loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1))
        loss = loss_fn(pred, target)
        print("LOSS I AM HERE", loss.item())
        #loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss, jac_loss, sradius]
        else:
            return [loss, jac_loss, sradius] + new_mems
