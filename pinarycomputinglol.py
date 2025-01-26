(* Initialize Parameters *)
A = 1; (* Divergent Value *)
W = 1; (* Convergent Value *)
N = Pi; (* Fixed Scaling Factor (Pi) *)
numIterations = 10; (* Number of Iterations in Feedback Loop *)
emoji = "\[HappySmiley]"; (* Example Emoji *)

(* Function to Simulate Observer Feedback *)
getFeedback[output_] := RandomReal[{-1, 1}]; (* Placeholder for feedback *)

(* Function for Pi-Nary Transformation *)
toPiNaryExtended[binarySeq_List, numDigits_: 15, scale_: 1] := 
 Module[{piDigits, positions, transformed},
  piDigits = RealDigits[Pi, 10, numDigits][[1]]; (* Extract Pi Digits *)
  positions = Position[piDigits, _?(# != 0 &), {1}, Heads -> False];
  transformed = 
   Table[
    If[MemberQ[Flatten[positions], i - 1], BitXor[binarySeq[[i]], 1], 
     binarySeq[[i]]], {i, 1, Length[binarySeq]}];
  transformed
 ];

(* Feedback Loop *)
Do[
  (* 1. Calculate U Based on Current A, W, and Fixed N (Pi) *)
  U = (A*10^N) + (W*10^(-N));

  (* 2. Generate Pi-Nary Encoded Output *)
  emojiBinary = 
   Flatten[IntegerDigits[ToCharacterCode[emoji, "UTF-8"], 2, 8]];
  piNaryData = toPiNaryExtended[emojiBinary, 15, U];
  output = 
   FromCharacterCode[FromDigits[#, 2] & /@ Partition[piNaryData, 8], 
    "UTF-8"];

  (* 3. Simulate Feedback and Update Parameters *)
  feedback = getFeedback[output];
  A = A + feedback*0.1; (* Adjust Divergent Value *)
  W = W - feedback*0.1; (* Adjust Convergent Value *)

  (* 4. Log Iteration Data *)
  Print[
   "Iteration: ", i, ", A: ", A, ", W: ", W, ", N: ", N, ", U: ", U, ", Output: ", 
   output];
  , {i, 1, numIterations}]

import numpy as np
import matplotlib.pyplot as plt

# Helix parameters
omega = 2 * np.pi  # Angular velocity
v = 1.0            # Translational velocity
t = np.linspace(0, 10, 500)  # Time steps

# Helical coordinates
x = np.cos(omega * t)
y = np.sin(omega * t)
z = v * t

# Raytracing and Raycasting
def raycasting(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def raytracing(x1, y1, z1):
    # Simple reflection: inversion transformation
    return 1 / np.sqrt(x1**2 + y1**2 + z1**2)

# Compute midpoint for each point
raycast_distances = raycasting(x[:-1], y[:-1], z[:-1], x[1:], y[1:], z[1:])
raytrace_distances = raytracing(x[:-1], y[:-1], z[:-1])

midpoints = 1 / (1 / raycast_distances + 1 / raytrace_distances)

# Plot results
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot(x, y, z, label="Helical Path")
ax.set_title("Helical Path")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax2 = fig.add_subplot(122)
ax2.plot(t[:-1], midpoints, label="Midpoint (r)")
ax2.set_title("Midpoint of Raycasting and Raytracing")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Midpoint (r)")
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
from scipy.optimize import linear_sum_assignment

# Fibonacci and Lucas sequences
def fibonacci(n):
    if n < 0:
        return fibonacci(n + 2) - fibonacci(n + 1)
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def lucas(n):
    if n < 0:
        return lucas(n + 2) - lucas(n + 1)
    elif n == 0:
        return 2
    elif n == 1:
        return 1
    else:
        return lucas(n - 1) + lucas(n - 2)

# Photon path modeling
def photon_cost(x, y, z, m, n):
    distances = np.sqrt((m[0] * np.diff(x))**2 + (m[1] * np.diff(y))**2 + (m[2] * np.diff(z))**2)
    costs = distances * np.array([fibonacci(n) + lucas(-n) for n in range(len(distances))])
    return costs.sum()

# Sample 3D points (nodes)
points = np.array([
    [0, 0, 0],
    [1, 2, 3],
    [4, 1, 0],
    [7, 5, 2]
])

# Coefficients for directionality
m = (1, 1, 1)

# Calculate total photon cost
x, y, z = points[:, 0], points[:, 1], points[:, 2]
n_steps = len(points) - 1
total_cost = photon_cost(x, y, z, m, n_steps)

print(f"Total Photon Cost: {total_cost:.4f}")

import numpy as np

# Define flow rates as functions of time
def f1(t):
    return np.sin(t) + 1  # Example flow rate for F1

def f2(t):
    return np.cos(t) + 1  # Example flow rate for F2

# Time parameters
t = np.linspace(0, 2 * np.pi, 1000)  # Time array
dt = t[1] - t[0]

# Compute cumulative Sigma for each flow
Sigma1 = np.cumsum(f1(t)) * dt
Sigma2 = np.cumsum(f2(t)) * dt

# Synchronization time (Sigma midpoint)
Sigma = (Sigma1 + Sigma2) / 2

# Plot results
import matplotlib.pyplot as plt

plt.plot(t, Sigma1, label="Sigma1 (F1 cumulative)")
plt.plot(t, Sigma2, label="Sigma2 (F2 cumulative)")
plt.plot(t, Sigma, label="Synchronized Sigma")
plt.title("Sigma Synchronization of Opposite Flows")
plt.xlabel("Time (t)")
plt.ylabel("Cumulative Sigma")
plt.legend()
plt.grid()
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

