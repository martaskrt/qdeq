import numpy as np
import qutip as qt
import sys
import os


import torch
import torchdiffeq

""" Helper functions for creating initial states of relevance to TFIMs"""


def initial_state_constructor(chain_size, coefficients=[1, 0]):  # Logical states of Ising model QEC code.
    state_0 = qt.basis(2, 0)
    state_1 = qt.basis(2, 1)
    for i in range(1, chain_size):
        state_0 = qt.tensor(state_0, qt.basis(2, 0))
        state_1 = qt.tensor(state_1, qt.basis(2, 1))
    return (coefficients[0] * state_0 + coefficients[1] * state_1).unit()


def domain_wall_creator(chain_size):  # Adding domain walls.
    system_size = chain_size
    return term_constructor(system_size, qt.sigmax(), [k for k in range(0, int(chain_size / 2))])


def random_initialization(chain_size, logical_state):  # Randomized initial state.
    initial_state = initial_state_constructor(chain_size, logical_state)
    for i in range(0, chain_size):
        randomizer = term_constructor(chain_size, qt.rand_unitary(2), [i])
        initial_state = randomizer*initial_state
    return initial_state


""" Debug and quality of life functions """


def silence():  # Makes mcsolve shut up.
    sys.stdout = open(os.devnull, 'w')


def restore_print():  # Re-enables print after shutting up mcsolve.
    sys.stdout = sys.__stdout__


def state_printer(state, cutoff=0.1):  # Helpful for printing main contributions of pure states.
    for i in range(0, state.shape[0]):
        if np.abs(state.data[i]) > cutoff:
            print(state.data[i], "| ", np.binary_repr(i), ">")
    return 0


""" Constructing actually relevant operators """


def term_constructor(system_size, operator, sites):  # Helper code to build local operators in multi-qubit space.
    if 0 in sites:
        H = operator
    else:
        H = qt.identity(2)
    for i in range(1, system_size):
        if i in sites:
            H = qt.tensor(H, operator)
        else:
            H = qt.tensor(H, qt.identity(2))
    return H


def TFIM_constructor(system_size, J=1, h=1):  # Building the full Hamiltonian.
    H = 0
    for i in range(0, system_size):
        H += h * term_constructor(system_size, qt.sigmax(), [i])
        H += J * term_constructor(system_size, qt.sigmaz(), [i, (i + 1) % system_size])
    return H


def noise_constructor(system_size, operator, rate=0.01):  # Helper code to add identical noise to all qubits.
    jump_operators = []
    for i in range(0, system_size):
        jump_operators.append(rate * term_constructor(system_size, operator, [i]))
    return jump_operators




""" Setting up the system """
print("Initializing system...")
chain_size = 7  # Number of qubits.
T = 273  # "Temperature" // room temperature superconductor
beta = 1/T
t = 0.1  # Timestep related to the thermalizing map.

H = TFIM_constructor(chain_size, h=0.5)  # Hamiltonian and noise is defined.
assert H.isherm
noise = (noise_constructor(chain_size, qt.sigmap(), rate=0.1*np.exp(-beta)) +
         noise_constructor(chain_size, qt.sigmam(), rate=0.1))
assert np.all(n.iscptp for n in noise)
# print(noise)

H_torch = torch.tensor(H.full(), dtype=torch.cfloat)
for n in noise:
    print(n)
noise_tensor_list = [torch.tensor(n.full(), dtype=torch.cfloat) for n in noise]


""" Running simulations """
print("Running simulations...")
zero_matrix = np.zeros((2**chain_size, 2**chain_size), dtype=np.complex64)  # Preparing an empty density matrix.
tlist = np.linspace(0, t, 100)
options = qt.Options(store_final_state=True)  # Asking it to not throw out the data we need.

for i in range(0, 2**chain_size):
    for j in range(0, 2**chain_size):

        '''
        state = zero_matrix  # Setting precisely one entry in the density matrix equal one.
        state[i][j] = 1

        state = qt.Qobj(state, dims=H.dims)  # Wrapping it in QuTiP dictionary.

        result = qt.mesolve(H, state, tlist, c_ops=noise, options=options)  # Running the actual simulation.


        np.save("./results/stuff_" + str(i) + "_" + str(j) + ".npy", result.final_state.data)  # Saving resulting data.
        '''
    print(i)  # I am an impatient man.



""" now do the same stuff in pytorch """
print(" now do the same stuff in pytorch ")
# transform Hamiltonian to torch.tensor

# print(ham_torch)
# print(noise_list)

L = qt.liouvillian(H, noise)

Liou = torch.tensor(L.full(), dtype=torch.cfloat)
dim = 2**chain_size

u = torch.linalg.eig(Liou)
print(u)

krausis = qt.to_kraus(L)
krausis_torch = [torch.tensor(kraus.full(), dtype=torch.cfloat) for kraus in krausis]



# state = qt.Qobj(state, dims=H.dims)  # Wrapping it in QuTiP dictionary.
# state = torch.tensor(state, dtype=torch.complex64)


def kraus_witch(kraus, state):
    assert kraus.shape == state.shape
    return torch.einsum('ij, jk, lk -> il', kraus, state, torch.conj(kraus))

def apply_liouvillian_via_kraus(krausis):

    def wrap_L_func(time, state):

        state = torch.reshape(state, (dim,dim))
        state_out = kraus_witch(krausis[0], state)
        for k in range(1, len(krausis)):
            state_out += kraus_witch(krausis[k], state)
            # state = torch.einsum('ij, jk, kl -> il', kraus, torch.reshape(state, kraus.shape), torch.conj(kraus.t()))
        # return state.flatten()
        return state_out.flatten()

    return wrap_L_func

def apply_liouvillian_via_superoperator(liouvillian):

    def wrap_L_func(time, state):
        # state = state.reshape((dim,dim))

        out = torch.einsum('mn, n -> m', liouvillian, state)
        # return out.flatten()
        return out

    return wrap_L_func

def our_own_euler_forward(L_func, tlist, initial_state):
    current_state = initial_state
    dt = tlist[1] - tlist[0]
    for jj in range(len(tlist)):
        current_state += dt*L_func(38, current_state) # 38 is the new 42

    return current_state


import torchdiffeq






# d/dt rho = L[rho]
"""
    odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None)
"""
def our_mesolve(L_func, initial_state, time_list):
    """ our def of mesolve """
    result = odeint(L_func, initial_state, time_list)
    return result

def prepare_input_state(dim):
    out = np.zeros((dim,dim), dtype=np.complex128)
    out[2][2] = 1.+0.j
    return out
""" solve stuff, but only for one input and not for 2**1000 """
# qutip thingies
# state[i][j] = 1/dim
state = prepare_input_state(dim)
import copy
state_torch = copy.deepcopy(state)

state = qt.Qobj(state, dims=H.dims)  # Wrapping it in QuTiP dictionary.


result = qt.mesolve(H, state, tlist, c_ops=noise, options=options)  # Running the actual simulation.

print(result.states[-1])
out_true = torch.tensor(result.states[-1].full())

def lindblad_dissipator(operator, state, gamma:float=1.):
    return gamma * (operator @ state @ operator.conj().T) - 0.5 * operator.conj().T @ operator @ state - 0.5 * state @ operator.conj().T @ operator

def lindblad_rhs(t, state, H, noise):
    state_matrix = state.reshape((dim, dim))
    rhs = torch.tensor([-1j], dtype=torch.cfloat) * (H @ state_matrix - state_matrix @ H)
    for n in noise:
        rhs += lindblad_dissipator(n, state_matrix)
    return rhs.flatten()

# pytroch thingies
state = torch.tensor(prepare_input_state(dim), dtype=torch.cfloat)
state = state.flatten()


# Solve the master equation
result = torchdiffeq.odeint(lambda t, y: lindblad_rhs(t, y, H_torch, noise_tensor_list), state, torch.tensor(tlist))


out21 = result[-1]
print(out21)

res_new = torch.real(out21) -1.j* torch.imag(out21)
print(torch.linalg.norm(out21-out_true.flatten()))
