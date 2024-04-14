import numpy as np

def apply_operator(operator, state):
    return np.dot(operator, state)

def lindblad_dissipator(operator, state, gamma):
    return gamma * (np.dot(operator, np.dot(state, operator.conj().T)) - 0.5 * np.dot(operator.conj().T.dot(operator), state) - 0.5 * np.dot(state, operator.conj().T.dot(operator)))

def evolve(state, hamiltonian, dissipators, dt):
    state_temp = state.copy()
    for dissipator in dissipators:
        state_temp += dissipator(state) * dt
    state_temp += -1j * (hamiltonian @ state_temp - state_temp @ hamiltonian) * dt
    return state_temp

def wrap_evolve(hamiltonian, dissipators, dt):
    def inner_evolve(time, state):
        state_temp = state.copy()
        for dissipator in dissipators:
           state_temp += dissipator(state) * dt
        state_temp += -1j * (hamiltonian @ state_temp - state_temp @ hamiltonian) * dt
        return state_temp
    return inner_evolve



# Parameters
omega = 1.0
gamma = 0.2
n_th = 0.5  # Thermal excitation number
t_max = 10
dt = 0.1
steps = int(t_max / dt)
state = np.array([[1, 0], [0, 0]], dtype=np.complex64)  # Initial density matrix

# Operators
a = np.array([[0, 0], [1, 0]], dtype=np.complex64)
a_dag = np.conj(a.T)
H = omega * (a_dag @ a)
dissipators = [lambda state: lindblad_dissipator(a, state, gamma * (1 + n_th)),
               lambda state: lindblad_dissipator(a_dag, state, gamma * n_th)]

# Time evolution
results = []
for step in range(steps):
    state = evolve(state, H, dissipators, dt)
    results.append(np.abs(state[0, 0]))  # Store probability of being in the ground state

print(results)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def apply_operator(operator, state):
    return np.dot(operator, state)

def lindblad_dissipator(operator, state, gamma):
    return gamma * (np.dot(operator, np.dot(state, operator.conj().T)) - 0.5 * np.dot(operator.conj().T.dot(operator), state) - 0.5 * np.dot(state, operator.conj().T.dot(operator)))

def lindblad_rhs(t, state, H, dissipators):
    state_matrix = state.reshape((2, 2))
    rhs = -1j * (H @ state_matrix - state_matrix @ H)
    for dissipator in dissipators:
        rhs += dissipator(state_matrix)
    return rhs.flatten()

# Parameters
omega = 1.0
gamma = 0.2
n_th = 0.5  # Thermal excitation number
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)
state0 = np.array([1, 0, 0, 0], dtype=np.complex64)  # Initial density matrix

# Operators
a = np.array([[0, 0], [1, 0]], dtype=np.complex64)
a_dag = np.conj(a.T)
H = omega * (a_dag @ a)
dissipators = [lambda state: lindblad_dissipator(a, state, gamma * (1 + n_th)),
               lambda state: lindblad_dissipator(a_dag, state, gamma * n_th)]

# Solve the master equation
result = solve_ivp(lambda t, y: lindblad_rhs(t, y, H, dissipators), t_span, state0, t_eval=t_eval, method='RK45')
out = result.y.T
print(out[:,0])
