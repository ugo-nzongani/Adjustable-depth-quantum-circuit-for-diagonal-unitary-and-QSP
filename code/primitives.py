import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister

def int_to_binary(k,n,reverse=True):
    """Converts an integer k to its binary representation on n bits
    Parameters
    ----------
    k : int
        Integer to convert
    n : int
        Number of bits on which k should be written
    reverse : bool
        If True the indices of the bits in the bit string of k are reversed, i.e. binary[0] corresponds to
        the least significant bit of k
        
    Returns
    -------
    str
    """
    binary = bin(k)[2:].zfill(n)
    if reverse:
        binary = binary[::-1]
    return binary

def binary_to_int(k,reverse=True):
    """Converts a binary integer to its decimal representation
    Parameters
    ----------
    k : str
        Binary integer to convert
    reverse : bool
        Set to False if k[0] corresponds to the least significant bit of k
        
    Returns
    -------
    str
    """
    decimal = 0
    if reverse:
        k = k[::-1]
    for i in range(len(k)):
        decimal += 2**i * int(k[i])
    return decimal

def generate_gray_code(n, binary=False):
    """Generates a gray list of 2**n integers
    Parameters
    ----------
    n : int
        Binary integer to convert
    binary : bool
        Set to False if the list coefficient should be integers instead of strings
        
    Returns
    -------
    str array or int array
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")

    if n == 0:
        return ['0'*n] if binary else [0]
    
    gray_list = ['0'*n, int_to_binary(1,n,reverse=False)] if binary else [0, 1]
    
    for i in range(1, n):
        mirror = 2**i
        if binary:
            gray_list += [bin(mirror + int(num, 2))[2:].zfill(n) for num in reversed(gray_list)]
        else:
            gray_list += [mirror + int(num) for num in reversed(gray_list)]
    
    return gray_list

def changing_index(b1,b2):
    """Returns the list of indices that differ between b1 and b2
    Parameters
    ----------
    b1 : str
        First bit string
    b2 : str
        Second bit string
        
    Returns
    -------
    int list
    """
    if len(b1) != len(b2):
        print('Bit strings must have the same size.')
    else:
        index_list = []
        for i in range(len(b1)):
            if b1[i] != b2[i]:
                index_list.append(i)
        return index_list

def get_n_blocks(m,n_ancilla):
    """Returns the number of blocks of m qubits to implement with n_ancilla and 1 main register
    Parameters
    ----------
    m : int
        Number of working qubits
    n_ancilla : int
        Number of ancilla qubits
    
    Returns
    -------
    int
    """
    return int(np.floor(n_ancilla/m)) + 1

def n_operator_per_block(n_rotations,n_blocks):
    """Generates a list whose value of index i contains the number of rotation gates to put on block i
    Parameters
    ----------
    n_rotations : int
        Number of rotation gate in the circuit
    n_blocks : int
        Number of block in the circuit
    Returns
    -------
    int array
    """
    n_operator = [0]*n_blocks
    index_block = 0
    for i in range(n_rotations):
        n_operator[index_block] += 1
        index_block = (index_block+1)%n_blocks
    return n_operator

def copy(reg1,reg2):
    """Quantum circuit whose first register copies its bit values into the second register
    Parameters
    ----------
    reg1 : qiskit.circuit.quantumregister.QuantumRegister
        Register of qubits being copied
    reg2 : qiskit.circuit.quantumregister.QuantumRegister
        Register of qubits receving the copies
    
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
    """
    qc = QuantumCircuit(reg1,reg2)
    if reg2.size >= reg1.size:
        blocks = int(np.floor(reg2.size/reg1.size))
        ancilla_available = reg2.size
        for i in range(int(np.floor(np.log2(blocks)))+1):
            for j in range(2**i):
                if ancilla_available - reg1.size >= 0:
                    for k in range(reg1.size):
                        if j == 0:
                            qc.cx(reg1[k],reg2[k + reg1.size*(2**i-1)])
                        elif k+reg1.size*(2**i-1)+reg1.size*j < reg2.size:
                            qc.cx(reg2[k+reg1.size*(j-1)],reg2[k+reg1.size*(2**i-1)+reg1.size*j])
                    ancilla_available -= reg1.size
    return qc

def first_one_bit(b):
    """Returns the index of the first bit equal to 1 in b
    Parameters
    ----------
    b : str
        Bit string

    Returns
    -------
    int
    """
    for i in range(len(b)):
        if b[i] == '1':
            return len(b)-i-1
    return -1
   
def other_one_bits(b):
    """Returns the list of indices where the bits of b are equal to 1 excluding the most significant one
    Parameters
    ----------
    b : str
        Bit string

    Returns
    -------
    int list
    """
    first = first_one_bit(b)
    other_index = []
    for i in range(first):
        if b[::-1][i] == '1':
            other_index.append(i)
    return other_index

def dyatic(x,n):
    """Returns the list of coefficient of the dyatic expansion of x up to order n
    Parameters
    ----------
    x : float
        Real number in [0,1]
    n : int
        Index for the truncation of the dyatic expansion of x

    Returns
    -------
    int list  
    """
    l = np.zeros((n))
    a = x
    for i in range(n):
        if a-1/2**(i+1) >= 0:
            l[i] = 1
            a = a-1/2**(i+1)
    return l

def bits_to_zero(b):
    """Generates a list containing the indices of the bits equal to one in bit string b
    Parameters
    ----------
    b : str
        Bit string
    
    Returns
    -------
    int list
    """
    index_list = []
    for i in range(len(b)):
        if b[i] == '0':
            index_list.append(i)
    return index_list
