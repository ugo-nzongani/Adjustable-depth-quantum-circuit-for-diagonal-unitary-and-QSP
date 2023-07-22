import numpy as np
from qiskit import *
from qiskit.tools.visualization import *
from qiskit.extensions import *
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.providers.aer import QasmSimulator

"""Adjustable-depth quantum circuit for diagonal operator and QSP"""

def l(k):
    """
    Parameters
    ----------
    k : int
        An index used in the computation of Q10

    Returns
    -------
    int
    """
    if k == 0:
        return 1
    else:
        return 2**(k-1) - 1 + l(k-1)
        
def q0(n,m,i,f):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator
    f : fun

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q0
    """
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    # Adding the quantum gates
    for k in range(M):
        current_state = i*M*2 + k*2
        x1 = bin(current_state)[2:].zfill(n)[::-1]
        x2 = bin(current_state + 1)[2:].zfill(n)[::-1]
        dyadic1, dyadic2 = 0, 0
        for j in range(n):
            dyadic1 += int(x1[n-j-1])/2**(j+1)
            dyadic2 += int(x2[n-j-1])/2**(j+1)
        unitary = np.array([[np.exp(1j*f(dyadic1)),0], [0,np.exp(1j*f(dyadic2))]])
        gate = UnitaryGate(unitary, label='C'+str(i*M+k))
        if k == 0:
            qc.append(gate.control(1), [b[k], q[k]])
        else:
            qc.append(gate.control(1), [b[k], s[k-1]])
    return qc
    
def q2(n,m):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q2
    """
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    # Adding the quantum gates
    # Q20
    for j in range(m-2 +1):
        #qc.barrier()
        for k in range(2**(m-j-1)-2 +1):
            qc.cnot(b[int(2**m -1-2**(j+1)*(1/2 +k))], b[2**m -1-k*2**(j+1)])
    # Q21
    for j in range(m-1+1):
        #qc.barrier()
        for k in range(2**j -1+1):
            if k == 0:
                qc.cswap(b[(k+1)*2**(m-j)-1], q[0], s[int(2**(m-j)*(1/2 +k))-1])
            else:
                qc.cswap(b[(k+1)*2**(m-j)-1], s[k*2**(m-j)-1], s[int(2**(m-j)*(1/2 +k))-1])        
        if j != m-1:
            #qc.barrier()
            for l in range(2**(j+1)-2+1):
                qc.cnot(b[int(2**m -1-2**(m-j-1)*(1/2 +l))], b[2**m -1- l*2**(m-j-1)])
    return qc
    
def q10(n,m):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q10 used to build Q1
    """
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    # Adding the quantum gates
    for j in range(m-2 +1):
        #qc.barrier()
        for k in range(j+1,m-1+1):
            if j == 0:
                qc.cnot(q[k+1],s[l(k)-1])
            else:
                qc.cnot(q[k+1],s[l(k)+2**j - 1 - 1])
                for l_prime in range(2**j -2+1):
                    qc.cnot(s[l(k)+l_prime-1],s[l(k)+l_prime+2**j-1])   
    return qc
    
def q11(n,m,i):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit Q11 used to build Q1
    """
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    #qc.barrier()
    
    ctrl_qubits = [q[k+1] for k in range(m,n-1)] # called alpha in the paper
    # Bits flips
    for k in range(m,n-1):
        # Function g
        if np.floor((i/2**(k-m)) % 2) == 0:
            qc.x(q[k+1])
    # (n-m)-Toffoli gate
    if m != n-1 :
        qc.mct(ctrl_qubits,b[0])
    else:
        qc.x(b[0])    
    # Controlled-swaps
    for j in range(m-1 +1):
        #qc.barrier()
        qc.cswap(q[j+1],b[0],b[2**j])
        for k in range(1,2**j -1 +1):
            qc.cswap(s[j+k-1-1],b[k],b[k+2**j])
    #qc.barrier()     
    return qc
    
def build_u_i(n,m,i,f):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    i : int
        Iteration index of U_i operator
    f : fun
    
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit U_i
    """
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    all_qubits = [i for i in q] + [i for i in s] + [i for i in b]
    #qc.barrier()
    # Q1
    qc = qc.compose(q10(n,m),all_qubits)
    qc = qc.compose(q11(n,m,i),all_qubits)
    qc = qc.compose(q10(n,m).inverse(),all_qubits)
    # Q2
    qc = qc.compose(q2(n,m),all_qubits)
    # Q0
    qc = qc.compose(q0(n,m,i,f),all_qubits)
    # Q2_dagger
    qc = qc.compose(q2(n,m).inverse(),all_qubits)
    # Q1_dagger
    qc = qc.compose(q10(n,m),all_qubits)
    qc = qc.compose(q11(n,m,i).inverse(),all_qubits)
    qc = qc.compose(q10(n,m).inverse(),all_qubits)
    #qc.barrier()
    return qc
    
def adjustable_depth(n,m,f):
    """
    Parameters
    ----------
    n : int
        The number of qubits encoding the position
    m : int
        The parameter of the circuit
    f : fun

    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit implementing the position-dependent coin operator followed by the shift operator
    """
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    all_qubits = [i for i in q] + [i for i in s] + [i for i in b]
    for i in range(2**(n-m-1)):
        qc = qc.compose(build_u_i(n,m,i,f),all_qubits)
    return qc

def adjustable_depth_diagonal(n,m,p,f):
    # m = 0,...,n-p-1
    while m > n-p-1:
        m -= 1
    # p = 0,...,n-1
    if p > n-1:
        print('Wrong value for p')
        return 0
    print('Adjustable-depth(n='+str(n)+',m='+str(m)+',p='+str(p)+')')
    M = 2**m
    q = QuantumRegister(n, name= 'q')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, s, b)
    
    qc.h(q)
    adj = adjustable_depth(n-p,m,f)
    all_qubits = [q[0]] + [q[i] for i in range(1,n-p)] + [i for i in s] + [i for i in b]
    
    qc = qc.compose(adj, all_qubits)
    
    return qc

def reordering(l,n,p):
    reordered_list = []
    r = 2**(n-p)
    for i in range(r):
        for j in range(2**p):
            reordered_list.append(l[i+j*r])
    return reordered_list

### Matrix representation ###
def diagonal_unitary(n,f):
    diag_unitary = np.identity(2**n,dtype=np.complex128)
    for i in range(2**n):
        x = bin(i)[2:].zfill(n)[::-1]
        dyadic = 0
        for j in range(n):
            dyadic += int(x[n-j-1])/2**(j+1)
        diag_unitary[i][i] = np.exp(1j*f(dyadic))
    return diag_unitary

def hadamard_state(n):
    state = np.ones(2**n,dtype=np.complex128)
    state /= np.linalg.norm(state)
    return state

def fidelity(psi1,psi2,n):
    F = 0+0j
    for i in range(2**n):
        F += np.conj(psi1[i])*psi2[i]
    return(abs(F)**2)