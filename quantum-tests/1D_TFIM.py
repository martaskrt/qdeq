import numpy as np
import qutip as qt
import sys
import os
import torch

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
chain_size = 4  # Number of qubits.
T = 50  # "Temperature"
beta = 1/T
t = 0.1  # Timestep related to the thermalizing map.

H = TFIM_constructor(chain_size, h=0.5)  # Hamiltonian and noise is defined.
noise = (noise_constructor(chain_size, qt.sigmap(), rate=0.1*np.exp(-beta)) +
         noise_constructor(chain_size, qt.sigmam(), rate=0.1))
print(type(noise))
print(type(noise_constructor(chain_size, qt.sigmap(), rate=0.1*np.exp(-beta))[0]))
print(torch.tensor(noise_constructor(chain_size, qt.sigmap(), rate=0.1*np.exp(-beta))[0].full()))























""" Running simulations """
print("Running simulations...")
zero_matrix = np.zeros((2**chain_size, 2**chain_size))  # Preparing an empty density matrix.
tlist = np.linspace(0, t, 100)
options = qt.Options(store_final_state=True)  # Asking it to not throw out the data we need.

for i in range(0, 2**chain_size):
    for j in range(0, 2**chain_size):

        state = zero_matrix  # Setting precisely one entry in the density matrix equal one.
        state[i][j] = 1

        state = qt.Qobj(state, dims=H.dims)  # Wrapping it in QuTiP dictionary.


        result = qt.mesolve(H, state, tlist, c_ops=noise, options=options)  # Running the actual simulation.






        np.save("./results/stuff_" + str(i) + "_" + str(j) + ".npy", result.final_state.data)  # Saving resulting data.
    print(i)  # I am an impatient man.
