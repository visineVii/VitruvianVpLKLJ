import numpy as np
from sympy import symbols, pi, diff, integrate
from numba import jit, prange
import matplotlib.pyplot as plt

# Define symbolic variables
D, Z, U, N = symbols('D Z U N')

# Function definition
f = -(U * Z**pi) / (D**pi * N)

# Derivatives
f_prime = diff(f, D)
print(f"Derivative of f(D, Z): {f_prime}")

# Indefinite Integral
f_integral = integrate(f, D)
print(f"Indefinite Integral of f(D, Z): {f_integral}")

# Numerical Implementation
@jit(nopython=True, parallel=True)
def calculate_function(U, Z, D, N):
    """Compute f(D, Z) across multiple values of D."""
    return -(U * (Z**np.pi)) / ((D**np.pi) * N)

@jit(nopython=True, parallel=True)
def calculate_derivative(U, Z, D, N):
    """Compute f'(D, Z) across multiple values of D."""
    return (np.pi * U * (Z**np.pi)) / ((D**(np.pi + 1)) * N)

# Parameters
U_val = 10
Z_val = 5
N_val = 2
D_vals = np.linspace(1, 100, 1000)  # Range of D values

# Calculate function and derivative values
f_vals = calculate_function(U_val, Z_val, D_vals, N_val)
f_prime_vals = calculate_derivative(U_val, Z_val, D_vals, N_val)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(D_vals, f_vals, label="f(D, Z)")
plt.plot(D_vals, f_prime_vals, label="f'(D, Z)", linestyle="--")
plt.xlabel("D")
plt.ylabel("Function Value")
plt.title("f(D, Z) and Its Derivative")
plt.legend()
plt.grid()
plt.show()
                     ┌──────────────┐
                     │ Input Signal │
                     └──────┬───────┘
                            │
           ┌─────────────Direction Flow───────────────┐
           ▼                                          ▼
  ┌───────────────┐                          ┌───────────────┐
  │ Harmonic Node │                          │ Harmonic Node │
  │    (ϕ-based)  │                          │    (π-based)  │
  └───────┬───────┘                          └───────┬───────┘
          │                                          │
          ▼                                          ▼
 ┌────────────────┐                        ┌────────────────┐
 │ Fourier Module │─────────Quadrature────▶ Quadrature Node │
 │   Decomposer   │        Modulation      │ (High-Freq Map)│
 └───────┬────────┘                        └───────┬────────┘
         │                                         │
         ▼                                         ▼
  ┌─────────────┐                         ┌─────────────┐
  │   Resonance │────Interference Align──▶│   Control    │
  │  Alignment  │                         │    System    │
  └─────────────┘                         └─────────────┘

┌───────────────────────────────┐
│ Dynamic Compression Algorithm │
├──────────────┬───────────────┤
│ Input Signal │ Threshold (k) │
├──────────────┴───────────────┤
│ If k > Threshold:            │
│   Compress Data Stream        │
│ Else:                         │
│   Maintain Signal Integrity   │
└───────────────────────────────┘
