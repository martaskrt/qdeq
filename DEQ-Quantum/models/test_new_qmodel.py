import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import sys
sys.path.append('./')

from qdeq_model_with_evolution import EquilibriumModel

import qutip as qt

def prepare_input_state_at_ij(dim: int, i: int=2, j: int=2):
    out = np.zeros((dim, dim), dtype=np.complex64)
    out[i][j] = 1.+0.j
    return out

def prepare_random_input_density(dim: int):
    off_diagonal = np.random.rand(dim, dim)
    off_diagonal = (off_diagonal + off_diagonal.T) / 2  # Symmetrize the matrix

    # Set diagonal elements to satisfy trace condition
    np.fill_diagonal(off_diagonal, 1 - np.sum(off_diagonal, axis=1) + off_diagonal.diagonal())

    # Normalize to ensure trace equal to 1
    rho = off_diagonal / np.trace(off_diagonal)

    return np.array(rho, dtype=np.complex64)

""" Experimenting """
model = EquilibriumModel()
# Looping over all elements to check equality


input_state = torch.tensor(prepare_random_input_density(model.dim))
print('dim', model.dim)
print('trace input', torch.trace(input_state))
out_torch = model.forward(x=input_state)
for i in range(len(model.tlist)):
    print('trace at j', model.tlist[i], torch.trace(out_torch[i]))
out_torch = out_torch[-1,:,:]
state_qt = prepare_random_input_density(model.dim)
state_qt = qt.Qobj(state_qt, dims=model.H.dims)
print('input qt trace', state_qt.tr())
tlist = np.array(model.tlist.detach())
options = qt.Options(store_final_state=True, method='bdf')  # Asking it to not throw out the data we need.
result_qt = qt.mesolve(model.H, state_qt, tlist, c_ops=model.noise, options=options)  # Running the actual simulation.
for state in result_qt.states:
    print('trace at j', state.tr())
# print('trace mesolve', result_qt.states[-1].tr())

print('our output: ', model.dim*out_torch)
print('qt output: ', model.dim*result_qt.states[-1])
out_true = torch.tensor(result_qt.states[-1].full())

print(torch.linalg.norm(out_torch-out_true, ord=1))
