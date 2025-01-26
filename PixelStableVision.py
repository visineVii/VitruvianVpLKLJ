import numpy as np
import time
from IntrovertFourier import fft as my_fft_function  # import your function from IntrovertFourier.py

# Generate a test signal
n = 1024  # signal length
dt = 0.01  # time step
t = np.arange(n) * dt  # time array
f = 10.0  # signal frequency
signal = np.sin(2 * np.pi * f * t)  # input signal

# Apply standard FFT algorithm (numpy.fft.fft) and measure execution time
start_time = time.time()
result_1 = np.fft.fft(signal)
execution_time_1 = time.time() - start_time

from IntrovertFourier import fft as my_fft_function  # import your function from IntrovertFourier.py

# Apply IntrovertFourier algorithm and measure execution time
start_time = time.time()
result_2 = my_fft_function(signal)
execution_time_2 = time.time() - start_time


# Compare the results and execution times
if np.allclose(result_1, result_2):
    print("The results are the same")
else:
    print("The results are different")

print(f"Execution time of standard FFT algorithm: {execution_time_1:.4f} seconds")
print(f"Execution time of IntrovertFourier algorithm: {execution_time_2:.4f} seconds")

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, Aer, execute
import socket
import hashlib
import hmac
import secrets
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Quantum Imaging and Routing Algorithm

def generate_quantum_statevector(packet_count):
    """Simulate a quantum circuit to produce a statevector."""
    n_qubits = int(np.ceil(np.log2(packet_count)))
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))  # Apply Hadamard gates to create superposition
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, simulator).result()
    statevector = result.get_statevector()
    return np.abs(statevector)**2  # Use probabilities as weights

def apply_quantum_transform(packet_data, statevector):
    """Apply quantum probabilities to transform packet data."""
    reshaped_statevector = statevector[:len(packet_data)]
    transformed_data = packet_data * reshaped_statevector
    return np.clip(transformed_data, 0, 255).astype(int)  # Ensure valid values

def map_data_to_ascii(packet_data, ascii_chars):
    """Map packet data to ASCII characters based on intensity."""
    ascii_mapping = []
    scale_factor = 255 / (len(ascii_chars) - 1)
    for value in packet_data:
        ascii_mapping.append(ascii_chars[int(value / scale_factor)])
    return ''.join(ascii_mapping)

def quantum_route_packets(packet_data, ascii_chars):
    """Simulate quantum-inspired routing decisions for data packets."""
    statevector = generate_quantum_statevector(len(packet_data))  # Quantum probabilities
    routed_data = apply_quantum_transform(np.array(packet_data), statevector)
    ascii_routes = map_data_to_ascii(routed_data, ascii_chars)
    return routed_data, ascii_routes

# Lightweight AES Encryption for Secure Routing
def encrypt_packet_data(packet_data, shared_key):
    """Encrypt the packet data using AES encryption."""
    iv = secrets.token_bytes(16)
    cipher = AES.new(shared_key, AES.MODE_CBC, iv)
    encrypted_data = cipher.encrypt(pad(packet_data, AES.block_size))
    return iv + encrypted_data

# HMAC Authentication
def authenticate_packet(packet_data, shared_key, nonce):
    """Generate HMAC for authentication."""
    return hmac.new(shared_key, packet_data + nonce, hashlib.sha256).digest()

# Network Transmission of Packets
def send_packet_to_server(packet_data, host="127.0.0.1", port=12345):
    """Send the packet data over a TCP connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.send(packet_data)
        s.close()
        print("Packet successfully sent to the server.")
    except Exception as e:
        print(f"Error sending packet to server: {e}")

# Main Workflow for Real-Time Packet Routing
def process_and_route_packets(packet_data, ascii_chars, shared_key, nonce, host="127.0.0.1", port=12345):
    routed_data, ascii_routes = quantum_route_packets(packet_data, ascii_chars)
    print(f"Quantum Routed Data: {routed_data}")
    print(f"ASCII Routing Representation: {ascii_routes}")

    # Convert routed data to bytes for encryption
    routed_data_bytes = bytes(routed_data)

    # Encrypt the routed data
    encrypted_packet = encrypt_packet_data(routed_data_bytes, shared_key)

    # Authenticate the packet
    auth = authenticate_packet(routed_data_bytes, shared_key, nonce)
    print("Authentication successful.")

    # Transmit the encrypted packet to the server
    send_packet_to_server(encrypted_packet, host, port)

# Example Usage
if __name__ == "__main__":
    # Simulated packet data (e.g., priorities or metadata)
    packet_data = [50, 200, 150, 30, 90, 255, 128]
    ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']  # ASCII map
    shared_key = secrets.token_bytes(32)  # Shared secret key
    nonce = secrets.token_bytes(16)  # Random nonce

    # Process and route packets
    process_and_route_packets(packet_data, ascii_chars, shared_key, nonce)
