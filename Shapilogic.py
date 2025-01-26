import hashlib
import sympy

def generate_large_prime_with_sha(input_string: str, bit_length: int = 2048) -> int:
    """
    Generate a large random prime number using SHA hash of the input string.

    Args:
        input_string (str): Input string to hash.
        bit_length (int): Desired bit length of the prime number.

    Returns:
        int: A large prime number.
    """
    # Create a hash from the input string
    sha_hash = hashlib.sha512(input_string.encode()).hexdigest()
    
    # Convert hash to an integer
    hash_int = int(sha_hash, 16)
    
    # Adjust the hash to match the desired bit length
    if hash_int.bit_length() > bit_length:
        hash_int = hash_int >> (hash_int.bit_length() - bit_length)
    else:
        hash_int = hash_int << (bit_length - hash_int.bit_length())
    
    # Find the next prime number greater than or equal to the hashed value
    large_prime = sympy.nextprime(hash_int)
    
    return large_prime

# Generate a 2048-bit random prime number using SHA hashing
input_string = "Generate a very large prime number using SHA"
large_prime = generate_large_prime_with_sha(input_string)

large_prime
