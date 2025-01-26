# Convert the decimal number 2.448592653589793 to base 16 and base 2
import numpy as np

def convert_to_base(decimal, base, precision=20):
    """
    Converts a decimal number to a specified base with given precision.
    """
    integer_part = int(decimal)
    fractional_part = decimal - integer_part
    base_result = []
    
    # Convert integer part to the base
    while integer_part > 0:
        base_result.append(int(integer_part % base))
        integer_part //= base
    base_result.reverse()
    
    # Convert fractional part to the base
    fractional_result = []
    while len(fractional_result) < precision:
        fractional_part *= base
        fractional_result.append(int(fractional_part))
        fractional_part -= int(fractional_part)
    
    return base_result, fractional_result

# Given number
decimal_number = 2.448592653589793

# Conversion to base 16
integer_base16, fractional_base16 = convert_to_base(decimal_number, 16)

# Conversion to base 2
integer_base2, fractional_base2 = convert_to_base(decimal_number, 2)

# Format the results
base16_result = f"{'.'.join([''.join(hex(x)[2:] for x in integer_base16)] + [''.join(hex(x)[2:] for x in fractional_base16)])}_16"
base2_result = f"{'.'.join([''.join(bin(x)[2:] for x in integer_base2)] + [''.join(bin(x)[2:] for x in fractional_base2)])}_2"

base16_result, base2_result


def reflect_8bit(byte):
    """
    Reflect an 8-bit byte by reversing the bit order.
    """
    return int(f"{byte:08b}"[::-1], 2)

def reflect_bytes(hex_value, endianness="big"):
    """
    Reflects all bytes in a hexadecimal value and converts between big and little endianness.
    """
    # Split hex into bytes
    byte_array = [int(hex_value[i:i+2], 16) for i in range(0, len(hex_value), 2)]
    # Reflect each byte
    reflected_bytes = [reflect_8bit(byte) for byte in byte_array]
    # Reorder bytes for specified endianness
    if endianness == "little":
        reflected_bytes.reverse()
    # Return as hex string
    return ''.join(f"{byte:02x}" for byte in reflected_bytes)

# Input hexadecimal values for IEEE formats
ieee_single_be = "401cb5be"
ieee_single_le = "beb51c40"
ieee_double_be = "400396b7bec326f3"
ieee_double_le = "f326c3beb7960340"
ieee_quad_be = "4000396b7bec326f369697e2c92f952c"
ieee_quad_le = "2c952fc9e29796366f32ec7b6b390040"

# Compute 8-bit reflections for all IEEE formats
results = {
    "IEEE Single-Precision BE": reflect_bytes(ieee_single_be, "big"),
    "IEEE Single-Precision LE": reflect_bytes(ieee_single_le, "little"),
    "IEEE Double-Precision BE": reflect_bytes(ieee_double_be, "big"),
    "IEEE Double-Precision LE": reflect_bytes(ieee_double_le, "little"),
    "IEEE Quad-Precision BE": reflect_bytes(ieee_quad_be, "big"),
    "IEEE Quad-Precision LE": reflect_bytes(ieee_quad_le, "little"),
}

results
