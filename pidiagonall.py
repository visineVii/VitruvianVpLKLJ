import numpy as np
import matplotlib.pyplot as plt

# Parameters
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
side_length = 0.5  # Half side length
pi_decimal = [int(d) for d in str(np.pi).replace('.', '')]  # Pi as decimals

# Generate the diagonal Pi Tile
def diagonal_pi_tile(n, steps=100):
    x = np.linspace(-side_length, side_length, steps)
    y = np.linspace(-side_length, side_length, steps)
    grid = np.zeros((steps, steps))

    for i in range(steps):
        for j in range(steps):
            diagonal = np.sqrt(x[i]**2 + y[j]**2)
            tile_value = (diagonal * (np.pi * phi**n)) % 10  # Modulo to encode decimals
            grid[i, j] = tile_value
    return grid

# Plot the diagonal Pi Tile
n = 1  # Transcendental step
tile = diagonal_pi_tile(n)

plt.figure(figsize=(8, 8))
plt.imshow(tile, cmap='viridis', extent=[-side_length, side_length, -side_length, side_length])
plt.colorbar(label='Pi Decimal Encodings')
plt.title("Diagonal Pi Tile with Universal Time Structure")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

import torch
import torch.nn as nn

class PiLogicNet(nn.Module):
    def __init__(self):
        super(PiLogicNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Example: 2 outputs for movement direction

    def forward(self, x):
        x = torch.sin(self.fc1(x))  # Harmonic activation
        x = torch.cos(self.fc2(x))  # Complementary harmonic layer
        x = self.fc3(x)
        return x

# Instantiate the model
model = PiLogicNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def parse_whitespace(code):
    """Parse whitespace into symbolic operations."""
    operations = {" ": "pi_operation", "\t": "i_transition", "\n": "reset"}
    return [operations[char] for char in code if char in operations]

# Example whitespace input
whitespace_code = "  \t\n "
parsed_operations = parse_whitespace(whitespace_code)
print(parsed_operations)
# Output: ['pi_operation', 'pi_operation', 'i_transition', 'reset']

from math import pi

def prioritize_task(task):
    return task["priority"] / (1 + task["deadline"])

sorted_tasks = sorted(tasks, key=prioritize_task)

tasks = [
    {"name": "Compute Pi", "priority": pi / 2},
    {"name": "Transition Task", "priority": abs(i)}
]

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]  # Create m slots, each initialized as an empty list

    def hash_function(self, key):
        """Simple hash function using modulo."""
        return key % self.size

    def insert(self, key, value):
        """Insert a key-value pair into the hash table."""
        index = self.hash_function(key)
        # Check if the key already exists and update it
        for pair in self.table[index]:
            if pair[0] == key:
                pair[1] = value
                return
        # If key doesn't exist, append the new key-value pair
        self.table[index].append([key, value])

    def search(self, key):
        """Search for a key in the hash table."""
        index = self.hash_function(key)
        for pair in self.table[index]:
            if pair[0] == key:
                return pair[1]  # Return the value
        return None  # Key not found

    def delete(self, key):
        """Delete a key-value pair from the hash table."""
        index = self.hash_function(key)
        for i, pair in enumerate(self.table[index]):
            if pair[0] == key:
                self.table[index].pop(i)
                return True
        return False  # Key not found

    def display(self):
        """Display the hash table."""
        for i, slot in enumerate(self.table):
            print(f"Slot {i}: {slot}")

# Example Usage
hash_table = HashTable(size=10)
hash_table.insert(49, "Value49")
hash_table.insert(86, "Value86")
hash_table.insert(52, "Value52")

print("Initial Hash Table:")
hash_table.display()

print("\nSearch for key 86:", hash_table.search(86))
hash_table.delete(86)
print("\nHash Table after deleting key 86:")
hash_table.display()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
from pyDHM import phaseCompensation, utilities
import numpy as np

# Define parameters
wavelength = 532e-9
dx, dy = 1.12e-6, 1.12e-6
s_values = np.arange(2, 10, 0.5)  # Range of s values to explore

# Load hologram (replace with your hologram data)
hologram = utilities.imageRead("hologram.png")

# Function to calculate entropy
def calculate_entropy(data):
    hist, _ = np.histogram(data, bins=256)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

best_entropy = float('inf')
best_reconstruction = None

for s in s_values:
    # Fully-compensated phase reconstruction
    reconstructed_phase = phaseCompensation.ERS(hologram, upper=True, wavelength=wavelength, dx=dx, dy=dy, s=s, step=0.2)
    
    # Calculate entropy
    current_entropy = calculate_entropy(np.abs(reconstructed_phase))

    # Update best reconstruction if entropy is lower
    if current_entropy < best_entropy:
        best_entropy = current_entropy
        best_reconstruction = reconstructed_phase

# Display the best reconstruction
utilities.imageShow(best_reconstruction, "Best Reconstructed Phase (Entropy Optimized)")
print(f"Best entropy: {best_entropy}")
# Define parameters
pi = np.pi
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
g2, g3 = 1.0, 1.0  # Invariants (example values)
H, U = 1.0, 1.0  # Scaling factors
Lambda, s, t = 1.0, 1.0, 1.0
psi, Psi = 1.0, 1.0  # Interaction terms
time = np.linspace(0, 10, 500)  # Time for evolution

def weierstrass_zeta(z, g2, g3):
    """Approximation of Weierstrass zeta function."""
    return z**-1 - sum([z**(2*k-1)/(2*k-1) * ((g2 if k == 2 else g3 if k == 3 else 0) / (20 if k == 2 else 28 if k == 3 else 1)) for k in range(2, 6)])

def transformation(axis, g2, g3, H, U, Lambda, s, t, psi, Psi):
    """Compute transformation for a single axis."""
    zeta_value = weierstrass_zeta(np.pi - axis, g2, g3)
    scaling = (Lambda * s * t) / (g2 * psi * Psi)
    return (H * U * zeta_value)**scaling

# Generate x, y, z values for three axes
x_values = transformation(np.linspace(0.1, 1.0, 500), g2, g3, H, U, Lambda, s, t, psi, Psi)
y_values = transformation(np.linspace(0.1, 1.0, 500), g2, g3, H, U, Lambda, s, t, psi, Psi)
z_values = transformation(np.linspace(0.1, 1.0, 500), g2, g3, H, U, Lambda, s, t, psi, Psi)

# 3D Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, label='Zeta Transformed Line')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Zeta-Transformed Line')
plt.legend()
plt.show()
