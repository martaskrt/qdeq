import numpy as np
import qutip as qt
import sys
import os


import torch
import torchdiffeq


class TFIMEvolutionModel:
    # TODO some of the staticmethod's can be un-staticed by making some self references

    """ Helper functions for creating initial states of relevance to TFIMs"""
    def initial_state_constructor(self, chain_size, coefficients=[1, 0]):  # Logical states of Ising model QEC code.
        state_0 = qt.basis(2, 0)
        state_1 = qt.basis(2, 1)
        for i in range(1, chain_size):
            state_0 = qt.tensor(state_0, qt.basis(2, 0))
            state_1 = qt.tensor(state_1, qt.basis(2, 1))
        return (coefficients[0] * state_0 + coefficients[1] * state_1).unit()

    def domain_wall_creator(self):  # Adding domain walls.
        system_size = self.chain_size
        return term_constructor(system_size, qt.sigmax(), [k for k in range(0, int(self.chain_size / 2))])

    def random_initialization(self, logical_state):  # Randomized initial state.
        initial_state = initial_state_constructor(self.chain_size, logical_state)
        for i in range(0, chain_size):
            randomizer = term_constructor(self.chain_size, qt.rand_unitary(2), [i])
            initial_state = randomizer*initial_state
        return initial_state

    """ Debug and quality of life functions """
    # TODO I think we can just get rid of these
    @staticmethod
    def silence():  # Makes mcsolve shut up.
        sys.stdout = open(os.devnull, 'w')

    @staticmethod
    def restore_print():  # Re-enables print after shutting up mcsolve.
        sys.stdout = sys.__stdout__

    @staticmethod
    def state_printer(state, cutoff=0.1):  # Helpful for printing main contributions of pure states.
        for i in range(0, state.shape[0]):
            if np.abs(state.data[i]) > cutoff:
                print(state.data[i], "| ", np.binary_repr(i), ">")
        return 0

    """ Constructing actually relevant operators """
    @staticmethod
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

    def TFIM_constructor(self, system_size, J=1, h=1):
        """ Building the full Hamiltonian """
        H = 0
        for i in range(0, system_size):
            H += h * self.term_constructor(system_size, qt.sigmax(), [i])
            H += J * self.term_constructor(system_size, qt.sigmaz(), [i, (i + 1) % system_size])
        return H

    def noise_constructor(self, system_size, operator, rate=0.01):
        """ Helper code to add identical noise to all qubits """
        jump_operators = []
        for i in range(0, system_size):
            jump_operators.append(rate * self.term_constructor(system_size, operator, [i]))
        return jump_operators

    @staticmethod
    def lindblad_dissipator(operator, state, gamma:float=1.):
        """ applied Lindbladian """
        return gamma * (operator @ state @ operator.conj().T) - 0.5 * operator.conj().T @ operator @ state - 0.5 * state @ operator.conj().T @ operator

    def lindblad_rhs(self, t, state, H, noise):
        """ Lindblad master equation rhs; d/dt rho = -i[H,rho] + L(rho) """
        state_matrix = state.reshape((self.dim, self.dim))
        rhs = torch.tensor([-1j], dtype=torch.cfloat) * (H @ state_matrix - state_matrix @ H)
        for n in noise:
            rhs += self.lindblad_dissipator(n, state_matrix)
        return rhs.flatten()

    """ Setting up the system """
    def __init__(self):
        # TODO some of these values will be input, some of them more like hyperparameters; we should discuss
        print("Initializing Transverse Field Ising Model...")
        self.chain_size = 3  # Number of qubits.
        self.dim = 2**self.chain_size  # Hilbert space dimension
        self.T = 273  # "Temperature" // room temperature superconductor
        self.beta = 1/self.T  # inverse "temperature"
        self.t = 0.1  # Timestep related to the thermalizing map.
        self.n_timesteps = 100  # Number of steps to use to reach t
        self.tlist = torch.linspace(0, self.t, self.n_timesteps)
        self.H = self.TFIM_constructor(self.chain_size, h=0.5)  # Hamiltonian and noise is defined.
        # TODO more hyperparameters hidden in the noise
        self.noise = (self.noise_constructor(self.chain_size, qt.sigmap(), rate=0.1*np.exp(-self.beta)) +
                      self.noise_constructor(self.chain_size, qt.sigmam(), rate=0.1))
        assert np.all(n.iscptp for n in self.noise)

        """ Convert operators to pytorch tensors """
        # TODO might be able to do nograd on them
        self.H_torch = torch.tensor(self.H.full(), dtype=torch.cfloat)
        self.noise_tensor_list = [torch.tensor(n.full(), dtype=torch.cfloat) for n in self.noise]
        """ Place holders for states """
        # TODO initial state might be part of __init__
        self.initial_state = None  # Input state to time evolution
        self.final_state =  None  # Final state of time evolution

    """ Running simulations """
    def forward(self):
        print("Running simulations...")
        # TODO we could also take the initial state as external input here and just output the output state
        # TODO instead of storing it in the model I guess, depends on what we need for the architecture
        initial_state = self.initial_state.flatten()

        # Solve the master equation
        func = lambda t, y: self.lindblad_rhs(t, y, self.H_torch, self.noise_tensor_list)
        result = torchdiffeq.odeint(func, initial_state, self.tlist)
        self.output_state = result[-1].reshape((self.dim, self.dim))  # TODO only return final state -- we may be able to modify odeint so that only the final one is outputted anyway? idk

# TODO replace this function with  input state from torchquantum output before
# TODO note that this torchquantum output now is _not_ measured but should also just be a (dim,) sized tensor
# TODO then, what we probably want is make this a (pure) density matrix, by doing an outer product on it
def prepare_input_state(dim: int, i: int=2, j: int=2):
    out = np.zeros((dim, dim), dtype=np.complex64)
    out[i][j] = 1.+0.j
    return out

""" Experimenting """
model = TFIMEvolutionModel()
# Looping over all elements to check equality
for i in range(model.dim):
    for j in range(model.dim):
        input_state = torch.tensor(prepare_input_state(model.dim, i, j), dtype=torch.cfloat)
        model.initial_state = input_state
        model.forward()
        out_torch = model.output_state

        """ Here is just code for verification """
        state_qt = prepare_input_state(model.dim, i, j)

        state_qt = qt.Qobj(state_qt, dims=model.H.dims)  # Wrapping it in QuTiP dictionary.

        tlist = np.array(model.tlist.detach())
        options = qt.Options(store_final_state=True)  # Asking it to not throw out the data we need.
        result_qt = qt.mesolve(model.H, state_qt, tlist, c_ops=model.noise, options=options)  # Running the actual simulation.

        # print('our output: ', out_torch)
        # print('lasse output: ', result_qt.states[-1])
        out_true = torch.tensor(result_qt.states[-1].full())

        print(torch.linalg.norm(out_torch-out_true))
