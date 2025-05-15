import math


def golomb_encode(n, m):
    """
    Encode a non-negative integer n using Golomb coding with parameter m.
    
    Args:
    n (int): The non-negative integer to encode
    m (int): The Golomb parameter (must be positive)
    
    Returns:
    str: The Golomb-encoded binary string
    """
    q = n // m
    r = n % m
    
    # Unary coding of q
    unary = '1' * q + '0'
    
    # Binary coding of r
    b = math.ceil(math.log2(m))
    if m & (m - 1) == 0:  
        binary = format(r, f'0{b}b')
    else:
        if r < 2**b - m:
            binary = format(r, f'0{b-1}b')
        else:
            binary = format(r + 2**b - m, f'0{b}b')
    
    return unary + binary

def golomb_decode(code, m):
    """
    Decode a Golomb-coded binary string with parameter m.
    
    Args:
    code (str): The Golomb-coded binary string
    m (int): The Golomb parameter (must be positive)
    
    Returns:
    int: The decoded non-negative integer
    """
    # Decode unary part
    q = 0
    for bit in code:
        if bit == '1':
            q += 1
        else:
            break
    
    # Decode binary part
    b = math.ceil(math.log2(m))
    if m & (m - 1) == 0: 
        r = int(code[q+1:q+1+b], 2)
    else:
        if len(code) - q - 1 == b - 1:
            r = int(code[q+1:], 2)
        else:
            r = int(code[q+1:], 2) - (2**b - m)
    
    return q * m + r

def demonstrate_golomb_coding(numbers, m):
    """
    Demonstrate Golomb coding by encoding and decoding a list of numbers.
    
    Args:
    numbers (list): List of non-negative integers to encode and decode
    m (int): The Golomb parameter (must be positive)
    """
    print(f"Demonstrating Golomb coding with m = {m}")
    print("-" * 50)
    
    for n in numbers:
        encoded = golomb_encode(n, m)
        decoded = golomb_decode(encoded, m)
        
        print(f"Number: {n}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Correct: {n == decoded}")
        print("-" * 50)


    
