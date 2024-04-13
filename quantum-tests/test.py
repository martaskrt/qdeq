'''
import torch
from torchdiffeq import odeint

# Define a simple differential equation
def func(y, t):
    return torch.tensor([-1j]) * y

# Initial condition with complex value
y0 = torch.tensor([1 + 1j], dtype=torch.complex64)

# Time points to solve the ODE
t = torch.linspace(0, 1, 10)

# Solve the ODE
y = odeint(func, y0, t)

print(y)



# guy posted
a = torch.tensor([[1,0],[0,1]], dtype=torch.cfloat)
y0 = torch.tensor([1, 0], dtype=torch.cfloat)
def f(t, y):
        return -1.j * torch.matmul(a, y)
t_list = torch.linspace(0, 1, 11)
s = odeint(f, y0, t_list)
print(s)
'''


import numpy as np
import scipy.linalg as la
import qutip as qt

# Step 1: Create the Liouvillian operator using QuTiP
# Define your Hamiltonian and collapse operators
H = qt.Qobj([[1, 0], [0, -1]])  # Example Hamiltonian
c_ops = [qt.Qobj([[0, 1], [0, 0]])]  # Example collapse operator

# Create the Liouvillian
liouvillian = qt.liouvillian(H, c_ops)
# Step 2: Convert the Liouvillian operator to a NumPy array
liouvillian_array = liouvillian.full()

# Step 3: Define your initial state as a NumPy array
initial_state_array = np.array([1, 0, 0, 0])  # Example initial state

print(liouvillian_array)
# Step 4: Apply the Liouvillian to the initial state using NumPy's matrix multiplication
evolved_state_array = np.dot(liouvillian_array, initial_state_array)

print("Evolved state:", evolved_state_array)
