import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
psi = (1 - np.sqrt(5)) / 2  # Silver ratio
alpha = 0.5
omega = 2 * np.pi  # Frequency of 1 Hz

# Define the Cubit Brane Fields function
def CubitBraneFields(n):
    """Returns the CBF number for the given index n."""
    return alpha * phi * np.pi * psi * (1 + n)

# Function f(x, y, z) with dynamic parameters
def f(x, y, z, phi, alpha, omega, t):
    """Calculates the dynamic function based on given parameters."""
    base = phi ** (x + y + z)
    dynamic = alpha * base * np.cos(omega * t)
    return base + dynamic

# Calculate Pi using Cubit Brane Fields
def calculate_pi_using_Cubit_Brane_Fields(n):
    """Approximates Pi using Cubit Brane Fields."""
    return sum(CubitBraneFields(i) for i in range(n))

# Parameters
x, y, z = 1, 2, 3  # Spatial dimensions
t_values = np.linspace(0, 10, 100)  # Time values for visualization
n_terms = 1000  # Number of terms for pi approximation

# Generate dynamic function values over time
results = [f(x, y, z, phi, alpha, omega, t) for t in t_values]

# Calculate Pi
calculated_pi = calculate_pi_using_Cubit_Brane_Fields(n_terms)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(t_values, results, label="f(x, y, z) over time", color="blue")
plt.axhline(y=calculated_pi, color="red", linestyle="--", label="Approximated Pi")
plt.xlabel("Time (t)")
plt.ylabel("f(x, y, z)")
plt.title("Dynamic Behavior of Cubit Brane Fields")
plt.legend()
plt.grid()
plt.show()

# Output results
print(f"Approximated value of Pi using Cubit Brane Fields: {calculated_pi:.6f}")


# Parameters
N = 5  # Number of nodes
m_i = 1.0  # Mass at each node
hbar = 1.0  # Planck constant reduced
omega_i = np.linspace(1.0, 2.0, N)  # Frequencies for each node
k_ij = 0.5  # Coupling constant between nodes
timesteps = 100  # Number of time steps
dt = 0.1  # Time step size

# Graph structure: Adjacency matrix
adj_matrix = np.zeros((N, N))
for i in range(N - 1):
    adj_matrix[i, i + 1] = adj_matrix[i + 1, i] = 1

# Initialize wavefunctions and potential
psi_real = np.random.uniform(-1, 1, N)  # Real part of wavefunctions
psi_imag = np.random.uniform(-1, 1, N)  # Imaginary part of wavefunctions
psi = psi_real + 1j * psi_imag

# Potential energy based on graph
def graph_potential(psi, adj_matrix, k_ij):
    potential = 0
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] != 0:
                potential += k_ij * abs(psi[i] - psi[j])**2
    return potential

# Kinetic energy
def kinetic_energy(psi, m_i, hbar):
    return np.sum((hbar**2 / (2 * m_i)) * np.abs(np.gradient(psi))**2)

# Total energy calculation
def total_energy(psi, adj_matrix, k_ij, m_i, hbar):
    kinetic = kinetic_energy(psi, m_i, hbar)
    potential = graph_potential(psi, adj_matrix, k_ij)
    oscillatory = 0.5 * m_i * np.sum(omega_i**2 * np.abs(psi)**2)
    return kinetic + potential + oscillatory

# Time evolution of wavefunction
def time_evolution(psi, adj_matrix, dt, hbar, k_ij, m_i):
    laplacian = -np.dot(adj_matrix, psi)
    dpsi_dt = (-1j * hbar / m_i) * laplacian
    return psi + dpsi_dt * dt

# Simulation
psi_over_time = [psi.copy()]
energies = [total_energy(psi, adj_matrix, k_ij, m_i, hbar)]

for t in range(timesteps):
    psi = time_evolution(psi, adj_matrix, dt, hbar, k_ij, m_i)
    psi_over_time.append(psi.copy())
    energies.append(total_energy(psi, adj_matrix, k_ij, m_i, hbar))

# Visualization: Energy conservation
plt.figure()
plt.plot(energies, label="Total Energy")
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.title("Conservation of Energy")
plt.legend()
plt.show()

# Visualization: Wavefunction dynamics on the graph
fig, ax = plt.subplots()
x_positions = np.arange(N)
line, = ax.plot(x_positions, np.abs(psi_over_time[0]), 'bo-', label="|ψ|")

def update(frame):
    line.set_ydata(np.abs(psi_over_time[frame]))
    return line,

ani = FuncAnimation(fig, update, frames=len(psi_over_time), interval=100, blit=True)
plt.title("Wavefunction Dynamics on Fixed Graph")
plt.xlabel("Node Index")
plt.ylabel("|ψ|")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
N = 5  # Number of nodes
m_i = 1.0  # Mass at each node
hbar = 1.0  # Planck constant reduced
omega_i = np.linspace(1.0, 2.0, N)  # Frequencies for each node
k_ij = 0.5  # Coupling constant between nodes
timesteps = 100  # Number of time steps
dt = 0.1  # Time step size

# Graph structure: Adjacency matrix
adj_matrix = np.zeros((N, N))
for i in range(N - 1):
    adj_matrix[i, i + 1] = adj_matrix[i + 1, i] = 1

# Initialize wavefunctions and potential
psi_real = np.random.uniform(-1, 1, N)  # Real part of wavefunctions
psi_imag = np.random.uniform(-1, 1, N)  # Imaginary part of wavefunctions
psi = psi_real + 1j * psi_imag

# Potential energy based on graph
def graph_potential(psi, adj_matrix, k_ij):
    potential = 0
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] != 0:
                potential += k_ij * abs(psi[i] - psi[j])**2
    return potential

# Kinetic energy
def kinetic_energy(psi, m_i, hbar):
    return np.sum((hbar**2 / (2 * m_i)) * np.abs(np.gradient(psi))**2)

# Total energy calculation
def total_energy(psi, adj_matrix, k_ij, m_i, hbar):
    kinetic = kinetic_energy(psi, m_i, hbar)
    potential = graph_potential(psi, adj_matrix, k_ij)
    oscillatory = 0.5 * m_i * np.sum(omega_i**2 * np.abs(psi)**2)
    return kinetic + potential + oscillatory

# Time evolution of wavefunction
def time_evolution(psi, adj_matrix, dt, hbar, k_ij, m_i):
    laplacian = -np.dot(adj_matrix, psi)
    dpsi_dt = (-1j * hbar / m_i) * laplacian
    return psi + dpsi_dt * dt

# Simulation
psi_over_time = [psi.copy()]
energies = [total_energy(psi, adj_matrix, k_ij, m_i, hbar)]

for t in range(timesteps):
    psi = time_evolution(psi, adj_matrix, dt, hbar, k_ij, m_i)
    psi_over_time.append(psi.copy())
    energies.append(total_energy(psi, adj_matrix, k_ij, m_i, hbar))

# Visualization: Energy conservation
plt.figure()
plt.plot(energies, label="Total Energy")
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.title("Conservation of Energy")
plt.legend()
plt.show()

# Visualization: Wavefunction dynamics on the graph
fig, ax = plt.subplots()
x_positions = np.arange(N)
line, = ax.plot(x_positions, np.abs(psi_over_time[0]), 'bo-', label="|ψ|")

def update(frame):
    line.set_ydata(np.abs(psi_over_time[frame]))
    return line,

ani = FuncAnimation(fig, update, frames=len(psi_over_time), interval=100, blit=True)
plt.title("Wavefunction Dynamics on Fixed Graph")
plt.xlabel("Node Index")
plt.ylabel("|ψ|")
plt.legend()
plt.show()
