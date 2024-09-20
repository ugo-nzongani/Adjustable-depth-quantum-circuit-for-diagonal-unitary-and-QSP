import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import RZGate, PhaseGate, MCPhaseGate
from primitives import *

def sequential_coeff(j,f,N):
    """j-th coefficient of the N-th sequential decomposition of f
    Parameters
    ----------
    j : int
        Order of the sequential coefficient
    f : function
        Function of one variable
    N : int
        Integer representing the number of points on which is computed the sequential coefficient
    Returns
    -------
    float
    """
    return f(j/N)

def sequential_operator(n,order,theta,nots=True):
    """Quantum circuit implementing the sequential operator of a given order
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    order : int
        Order of the sequential operator
    theta : float
        sequential coefficient
    nots : bool
        Set to False to only put the rotation gates without the NOTs
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
    """
    q = QuantumRegister(n,name='q')
    qc = QuantumCircuit(q)
    if order%2 == 0:
        binary = int_to_binary(order,n,reverse=False)
    else:
        binary = int_to_binary(order-1,n,reverse=False)
    if nots:
        if order%2 == 0:
            qc.x(q[0])
        for i in range(0,n-1):
            if binary[i] == '0':
                qc.x(q[n-i-1])
    if n > 1:
        gate = MCPhaseGate(theta,n-1,label='$R_{'+str(order)+'}$')
    else:
        gate = PhaseGate(theta,label='$R_{'+str(order)+'}$')
    qc.append(gate,qc.qubits)
    if nots:
        if order%2 == 0:
            qc.x(q[0])
        for i in range(0,n-1):
            if binary[i] == '0':
                qc.x(q[n-i-1])
    return qc

def sequential_informations(n,list_operator_to_implement,f,gray_code=True):
    """Returns a dictionnary whose keys are the order of sequential operators to implement and the
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
    sequential_dict = {}
    n_operator_to_implement = len(list_operator_to_implement)
    if gray_code:
        gray_list = generate_gray_code(n)
        for i in gray_list:
            if i in list_operator_to_implement:
                sequential_dict[i] = sequential_coeff(i,f,2**n)
    else:
        for i in range(n_operator_to_implement): 
            sequential_dict[list_operator_to_implement[i]] = sequential_coeff(list_operator_to_implement[i],f,2**n)
    return sequential_dict

def sequential_circuit(n,f,sequential_info,gray_code=True):
    """Generates the quantum circuit implementing the sequential decomposition of function f with the sequential operators
        whose order are contain in sequential_info 
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
    sequential_operators_list = []
    order_list = list(sequential_info.keys())
    for order in order_list:
        if gray_code:
            sequential_operators_list.append(sequential_operator(n,order,sequential_info[order],nots=False))
        else:
            sequential_operators_list.append(sequential_operator(n,order,sequential_info[order]))
    for index,operator in enumerate(sequential_operators_list):
        if gray_code:
            # avoid putting the useless cnots
            if index == 0:
                nots_index = bits_to_zero(int_to_binary(order_list[index],n,reverse=True))
                for j in nots_index:
                    qc.x(q[j])
                qc.append(operator,qc.qubits)
            elif index > 0:
                current_index = int_to_binary(order_list[index],n,reverse=True)
                previous_index = int_to_binary(order_list[index-1],n,reverse=True)
                nots_index = changing_index(current_index,previous_index)
                for j in nots_index:
                    qc.x(q[j])
                qc.append(operator,qc.qubits)
            if index == len(sequential_operators_list)-1:
                nots_index = bits_to_zero(int_to_binary(order_list[index],n,reverse=True))
                for j in nots_index:
                    qc.x(q[j])
        else:
            qc.append(operator,qc.qubits)
    return qc
