import numpy as np

def pi_logic_cmyk_mirror(c, m, y, k, k_internal):
    # Pi constant
    pi = np.pi

    # Transform CMYK values
    c_prime = k_internal * (1 - m) * np.sin(pi * c)
    m_prime = k_internal * (1 - y) * np.cos(pi * m)
    y_prime = (1 / (k_internal * (1 + c))) * np.tan(pi * y)
    k_prime = k_internal * k * np.sqrt(pi * k)

    # Ensure values are within 0-1 range
    return np.clip([c_prime, m_prime, y_prime, k_prime], 0, 1)

# Example usage
c, m, y, k = 0.5, 0.6, 0.3, 0.8  # Input CMYK values
k_internal = 1.2  # Luke's middle "K"

c_prime, m_prime, y_prime, k_prime = pi_logic_cmyk_mirror(c, m, y, k, k_internal)

print(f"Transformed CMYK: C={c_prime:.2f}, M={m_prime:.2f}, Y={y_prime:.2f}, K={k_prime:.2f}")
