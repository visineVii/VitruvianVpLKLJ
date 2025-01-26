import numpy as np
from PIL import Image
import socket
import hashlib
import hmac
import secrets
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import os

# Shared secret key between client and server
shared_key = "my_shared_secret_key".encode()

# Generate a random nonce for secure communication
nonce = secrets.token_bytes(16)

# Define a function to compute Pi Logic inspired compression
def pi_logic_compress(image_path, chunk_size=8):
    """
    Compresses an image using Pi Logic inspired techniques.

    Args:
        image_path (str): Path to the image file.
        chunk_size (int): Size of image chunks for processing.

    Returns:
        compressed_data (bytes): Compressed image data in binary format.
    """

    # Load and preprocess the image (resize, grayscale)
    image = Image.open(image_path).convert('L')
    width, height = image.size
    image = image.resize((width // chunk_size * chunk_size, height // chunk_size * chunk_size))

    # Convert to NumPy array
    pixels = np.array(image)

    # Apply Pi-inspired transformations (example: modular arithmetic)
    transformed_pixels = np.mod(pixels, np.pi)  # Pi Logic-inspired transformation

    # Quantize pixel values (lossy compression)
    quantized_pixels = np.round(transformed_pixels / 255 * 16) * 16

    # Convert to binary representation
    binary_data = np.packbits(quantized_pixels)

    return binary_data

# Authentication and HMAC to ensure secure communication
def authenticate_data(data, shared_key, nonce):
    """Generate HMAC and compare with server's response for authentication."""
    return hmac.new(shared_key, data + nonce, hashlib.sha256).digest()

# AES encryption function
def encrypt_message(message, encryption_key):
    """Encrypt the message using AES with CBC mode."""
    iv = secrets.token_bytes(16)  # Generate a random IV
    cipher = AES.new(encryption_key, AES.MODE_CBC, iv)
    encrypted_message = cipher.encrypt(pad(message, AES.block_size))
    return iv + encrypted_message  # Return IV and ciphertext combined

# Send encrypted message to server
def send_message_to_server(encrypted_message):
    """Send the encrypted image data over a TCP connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 12345))  # Connect to server
    s.send(encrypted_message)  # Send encrypted data
    s.close()  # Close the socket

# Example usage: Image compression, encryption, and sending over TCP
def process_and_send_image(image_path):
    # Compress the image using Pi Logic-inspired compression
    compressed_data = pi_logic_compress(image_path)

    # Generate a random encryption key for AES encryption
    encryption_key = secrets.token_bytes(32)

    # Encrypt the compressed image data
    encrypted_message = encrypt_message(compressed_data, encryption_key)

    # Authenticate the encrypted message using HMAC
    auth = authenticate_data(compressed_data, shared_key, nonce)

    # Verify the HMAC response from the server (just an example, you'd typically verify the server response here)
    print("Authentication successful!")

    # Send the encrypted message to the server over TCP
    send_message_to_server(encrypted_message)

# Run the example with an image file
process_and_send_image("image.jpg")
