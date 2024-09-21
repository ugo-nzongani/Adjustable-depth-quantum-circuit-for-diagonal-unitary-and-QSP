import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import RZGate, PhaseGate, MCPhaseGate
from code.primitives import *

def walsh(j,x): 
    """The value of j-th Walsh function at position x
    Parameters
    ----------
    j : int
        Order of the Walsh function
    x : float
        Real number in [0,1]

    Returns
    -------
    float
    """
    jbin = bin(j)
    lj = len(jbin)-2
    X = dyatic(x,lj)
    p = 0
    for i in range(lj):
        p += int(int(jbin[-1-i]) * X[i])
    return (-1)**p

def walsh_coeff(j,f,N):
    """j-th Walsh coefficient of the N-th Walsh series of f
    Parameters
    ----------
    j : int
        Order of the Walsh coefficient
    f : function
        Function of one variable
    N : int
        Integer representing the number of points on which is computed the Walsh coefficient
    Returns
    -------
    float
    """
    k = np.array(range(N))/N
    a = 0
    for i in range(N):
        a += f(k[i]) * walsh(j,k[i])/N
    return a

def walsh_operator(n,order,walsh_coeff,cnots=True):
    """Quantum circuit implementing the Walsh operator of a given order
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    order : int
        Order of the Walsh operator
    walsh_coeffs : float
        Walsh coefficient
    cnots : bool
        Set to False to only put the rotation gates without the CNOTs
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
    """
    binary = int_to_binary(order,n,reverse=False)
    q = QuantumRegister(n,name='q')
    qc = QuantumCircuit(q)
    if order == 0:
        qc.p(walsh_coeff,q[0])
        qc.x(q[0])
        qc.p(walsh_coeff,q[0])
        qc.x(q[0])
    else:
        rotation_index = first_one_bit(binary)
        cnots_index = other_one_bits(binary)
        if cnots:
            for i in cnots_index:
                qc.cx(q[i],q[rotation_index])
        qc.append(RZGate(-2*walsh_coeff, label='$R_{'+str(order)+'}$'),[q[rotation_index]]) 
        if cnots:
            for i in cnots_index:
                qc.cx(q[i],q[rotation_index])          
    return qc

def walsh_informations(n,list_operator_to_implement,f,gray_code=True):
    """Returns a dictionnary whose keys are the order of Walsh operators to implement and the
    values are the associated coefficients
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    list_operator_to_implement : int list
        List of orders of the Walsh operators to implement
    f : function
        Function of one variable
    gray_code : bool
        Set to True if to get a gray code ordering of the orders of the Walsh operators
    Returns
    -------
    dict
    """
    walsh_dict = {}
    n_operator_to_implement = len(list_operator_to_implement)
    if gray_code:
        gray_list = generate_gray_code(n)
        for i in gray_list:
            if i in list_operator_to_implement:
                walsh_dict[i] = walsh_coeff(i,f,2**n)
    else:          
        for i in range(n_operator_to_implement):
            walsh_dict[list_operator_to_implement[i]] = walsh_coeff(list_operator_to_implement[i],f,2**n)
    return walsh_dict

def walsh_circuit(n,f,walsh_info,gray_code=True):
    """Generates the quantum circuit implementing the Walsh decomposition of function f with the Walsh operators
        whose order are contain in walsh_info
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    f : function
        Function of one variable
    walsh_info : dict
        Output of function walsh_informations, it is a dictionnary whose keys are the order of Walsh functions to implement and the
        values are the associated coefficients
    gray_code : bool
        Set to True if to get a gray code ordering of the orders of the Walsh operators
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
    """
    q = QuantumRegister(n,name='q')
    qc = QuantumCircuit(q)
    walsh_operators_list = []
    order_list = list(walsh_info.keys())
    for order in order_list:
        if gray_code:
            walsh_operators_list.append(walsh_operator(n,order,walsh_info[order],cnots=False))
        else:
            walsh_operators_list.append(walsh_operator(n,order,walsh_info[order]))
    for index,operator in enumerate(walsh_operators_list):
        if gray_code:
            # avoid putting the useless cnots
            rotation_index = first_one_bit(int_to_binary(order_list[index],n,reverse=False))
            if index == 0:
                cnots_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                for j in cnots_index:
                    qc.cx(q[j],q[rotation_index])
                qc.append(operator,qc.qubits)
            elif index > 0:
                previous_control_index = other_one_bits(int_to_binary(order_list[index-1],n,reverse=False))
                current_control_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                previous_rotation_index = first_one_bit(int_to_binary(order_list[index-1],n,reverse=False))
                previous_cnots = []
                current_cnots = []
                for i in previous_control_index:
                    previous_cnots.append((i,previous_rotation_index))
                for i in current_control_index:
                    current_cnots.append((i,rotation_index))
                cnots_index = list((set(previous_cnots)|set(current_cnots)) - set.intersection(set(previous_cnots),set(current_cnots)))
                for j in cnots_index:
                    qc.cx(q[j[0]],q[j[1]])
                qc.append(operator,qc.qubits)
            if index == len(walsh_operators_list)-1:
                cnots_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                for j in cnots_index:
                    qc.cx(q[j],q[rotation_index]) 
        else:
            qc.append(operator,qc.qubits)
    return qc
