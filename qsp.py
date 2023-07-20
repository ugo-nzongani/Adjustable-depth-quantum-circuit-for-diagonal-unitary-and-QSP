import numpy as np
from qiskit import *
import math
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
        
def q0_controlled(n,m,i,f,ancilla_qA):
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
    qA = AncillaRegister(ancilla_qA, name= 'qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    # Adding the quantum gates
    control_index = 0
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
            qc.append(gate.control(2), [b[k], qA[control_index], q[k]])
        else:
            qc.append(gate.control(2), [b[k], qA[control_index], s[k-1]])
        control_index += 1
    return qc
    
def q2_controlled(n,m,ancilla_qA):
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
    qA = AncillaRegister(ancilla_qA, name='qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    # Adding the quantum gates
    # Q20
    ccswap_gate = ccswap()
    for j in range(m-2 +1):
        #qc.barrier()
        control_index = 0
        for k in range(2**(m-j-1)-2 +1):
            qc.toffoli(b[int(2**m -1-2**(j+1)*(1/2 +k))], qA[control_index], b[2**m -1-k*2**(j+1)])
            control_index += 1
    # Q21
    for j in range(m-1+1):
        #qc.barrier()
        control_index = 0
        for k in range(2**j -1+1):
            if k == 0:
                qc.append(ccswap_gate, [b[(k+1)*2**(m-j)-1], qA[control_index], q[0], s[int(2**(m-j)*(1/2 +k))-1]])
            else:
                qc.append(ccswap_gate, [b[(k+1)*2**(m-j)-1], qA[control_index], s[k*2**(m-j)-1], s[int(2**(m-j)*(1/2 +k))-1]])
            control_index += 1
        if j != m-1:
            #qc.barrier()
            for l in range(2**(j+1)-2+1):
                qc.toffoli(b[int(2**m -1-2**(m-j-1)*(1/2 +l))], qA[control_index], b[2**m -1- l*2**(m-j-1)])
                control_index += 1
    return qc
    
def q10_controlled(n,m,ancilla_qA):
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
    qA = AncillaRegister(ancilla_qA,name='qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    # Adding the quantum gates
    for j in range(m-2 +1):
        #qc.barrier()
        control_index = 0
        for k in range(j+1,m-1+1):
            if j == 0:
                qc.toffoli(q[k+1],qA[control_index],s[l(k)-1])
            else:
                qc.toffoli(q[k+1],qA[control_index],s[l(k)+2**j - 1 - 1])
            control_index += 1
            for l_prime in range(2**j -2+1):
                qc.toffoli(s[l(k)+l_prime-1],qA[control_index],s[l(k)+l_prime+2**j-1])
                control_index += 1
    return qc
    
def q11_controlled(n,m,i,ancilla_qA):
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
    qA = AncillaRegister(ancilla_qA,name='qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    #qc.barrier()
    ccswap_gate = ccswap()
    ctrl_qubits = [q[k+1] for k in range(m,n-1)] + [qA[0]]
    # Bits flips
    control_index = 0
    for k in range(m,n-1):
        # Function g
        if np.floor((i/2**(k-m)) % 2) == 0:
            qc.cnot(qA[control_index],q[k+1])
            control_index += 1
    # (n-m)-Toffoli gate
    if m != n-1 :
        qc.mct(ctrl_qubits,b[0])
    else:
        qc.cnot(qA[0],b[0])    
    # Controlled-swaps
    for j in range(m-1 +1):
        #qc.barrier()
        qc.append(ccswap_gate, [q[j+1],qA[0],b[0],b[2**j]])
        control_index = 1
        for k in range(1,2**j -1 +1):
            qc.append(ccswap_gate, [s[j+k-1-1],qA[control_index],b[k],b[k+2**j]])
            control_index += 1
    #qc.barrier()     
    return qc
    
def ccswap():
    q = QuantumRegister(3)
    qc = QuantumCircuit(q)
    qc.cswap(q[0],q[1],q[2])
    gate = qc.to_gate().control(1)
    return gate
    
def build_u_i_controlled(n,m,i,f,ancilla_qA):
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
    qA = AncillaRegister(ancilla_qA,name='qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    all_qubits = [i for i in q] + [i for i in qA] + [i for i in s] + [i for i in b]
    #qc.barrier()
    # Q1
    qc = qc.compose(q10_controlled(n,m,ancilla_qA),all_qubits)
    qc = qc.compose(q11_controlled(n,m,i,ancilla_qA),all_qubits)
    qc = qc.compose(q10_controlled(n,m,ancilla_qA).inverse(),all_qubits)
    # Q2
    qc = qc.compose(q2_controlled(n,m,ancilla_qA),all_qubits)
    # Q0
    qc = qc.compose(q0_controlled(n,m,i,f,ancilla_qA),all_qubits)
    # Q2_dagger
    qc = qc.compose(q2_controlled(n,m,ancilla_qA).inverse(),all_qubits)
    # Q1_dagger
    qc = qc.compose(q10_controlled(n,m,ancilla_qA),all_qubits)
    qc = qc.compose(q11_controlled(n,m,i,ancilla_qA).inverse(),all_qubits)
    qc = qc.compose(q10_controlled(n,m,ancilla_qA).inverse(),all_qubits)
    #qc.barrier()
    return qc
    
def adjustable_depth_controlled(n,m,f,ancilla_qA):
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
    qA = AncillaRegister(ancilla_qA, name='qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    all_qubits = [i for i in q] + [i for i in qA] + [i for i in s] + [i for i in b]
    for i in range(2**(n-m-1)):
        qc = qc.compose(build_u_i_controlled(n,m,i,f,ancilla_qA),all_qubits)
    return qc

def adjustable_depth_diagonal_controlled(n,m,p,f):
    # m = 0,...,n-p-1
    while m > n-p-1:
        m -= 1
    # p = 0,...,n-1
    if p > n-1:
        print('Wrong value for p')
        return 0
    print('Adjustable-depth(n='+str(n)+',m='+str(m)+',p='+str(p)+')')
    
    # The largest number of operation to parallelize is:
    # 2**m 2x2 gates in Q0 or n-p-1 for the bit flips of Q11
    M = 2**m
    ancilla_qA = max(M, n-p-1)
    q = QuantumRegister(n, name= 'q')
    qA = AncillaRegister(ancilla_qA, name='qA')
    s = AncillaRegister(M-1, name= 's')
    b = AncillaRegister(M, name= 'b')
    qc = QuantumCircuit(q, qA, s, b)
    #qc.h(q)
    
    adj = adjustable_depth_controlled(n-p,m,f,ancilla_qA)
    all_qubits = [q[0]] + [q[i] for i in range(1,n-p)] + [i for i in qA] + [i for i in s] + [i for i in b]
    
    log_ancilla = math.ceil(np.log2(ancilla_qA))
    for i in range(log_ancilla):
        for j in range(2**i):
            qc.cnot(qA[j], qA[j+2**i])
            
    qc = qc.compose(adj, all_qubits)
    
    for i in range(log_ancilla):
        for j in range(2**(log_ancilla-i-1)):
            qc.cnot(qA[2**(log_ancilla-i-1)-j-1], qA[2**(log_ancilla-i-1)-j-1+2**(log_ancilla-i-1)])
    
    return qc