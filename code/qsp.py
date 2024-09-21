from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.library import MCXGate, CRZGate, CPhaseGate
from code.walsh import *
from code.sequential import *
from code.primitives import *

### WALSH ###

def walsh_operator_qsp(n,order,walsh_coeff,cnots=True):
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
    q = QuantumRegister(n+1,name='q')
    qc = QuantumCircuit(q)
    if order == 0:
        ctrl_phase_gate = CPhaseGate(walsh_coeff)
        qc.append(ctrl_phase_gate,[q[n],q[0]])
        qc.x(q[0])
        qc.append(ctrl_phase_gate,[q[n],q[0]])
        qc.x(q[0])
    else:
        rotation_index = first_one_bit(binary)
        cnots_index = other_one_bits(binary)
        if cnots:
            for i in cnots_index:
                qc.cx(q[i],q[rotation_index])
        qc.append(CRZGate(-2*walsh_coeff, label='$R_{'+str(order)+'}$'),[q[n],q[rotation_index]]) 
        if cnots:
            for i in cnots_index:
                qc.cx(q[i],q[rotation_index])          
    return qc

def walsh_circuit_qsp(n,n_ancilla_qsp,f,walsh_info,qsp_control_index,gray_code=True):
    """Generates the quantum circuit implementing the Walsh decomposition of function f with the Walsh operators
        whose orders are contain in walsh_info
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    n_ancilla_qsp : int
        Number of ancilla qubits used to control the rotations gates
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
    a_qsp = AncillaRegister(n_ancilla_qsp+1,name='a_qsp')
    qc = QuantumCircuit(q,a_qsp)
    
    q_qubits = qc.qubits[:n]
    a_qsp_qubits = qc.qubits[n:]
    
    walsh_operators_list = []
    order_list = list(walsh_info.keys())
    for order in order_list:
        if gray_code:
            walsh_operators_list.append(walsh_operator_qsp(n,order,walsh_info[order],cnots=False))
        else:
            walsh_operators_list.append(walsh_operator_qsp(n,order,walsh_info[order]))
    #qsp_control_index = 0
    for index,operator in enumerate(walsh_operators_list):
        if gray_code:
            # avoid putting the useless cnots
            rotation_index = first_one_bit(int_to_binary(order_list[index],n,reverse=False))
            if index == 0:
                cnots_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                for j in cnots_index:
                    qc.cx(q[j],q[rotation_index])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
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
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
            if index == len(walsh_operators_list)-1:
                cnots_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                for j in cnots_index:
                    qc.cx(q[j],q[rotation_index]) 
        else:
            qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
        #qsp_control_index = (qsp_control_index+1)%(n_ancilla_qsp+1)
    return qc
    
### SEQUENTIAL ###

def sequential_operator_qsp(n,order,theta,nots=True):
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
    q = QuantumRegister(n+1,name='q')
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
    gate = MCPhaseGate(theta,n,label='$R_{'+str(order)+'}$')
    qc.append(gate,qc.qubits)
    if nots:
        if order%2 == 0:
            qc.x(q[0])
        for i in range(0,n-1):
            if binary[i] == '0':
                qc.x(q[n-i-1])
    return qc

def sequential_circuit_qsp(n,n_ancilla_qsp,f,sequential_info,gray_code=True):
    """Generates the quantum circuit implementing the Sequential decomposition of function f with the Sequential operators
        whose orders are contain in sequential_info
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    n_ancilla_qsp : int
        Number of ancilla qubits used to control the rotations gates
    f : function
        Function of one variable
    sequential_info : dict
        Output of function sequential_informations, it is a dictionnary whose keys are the order of the sequential operator
        to implement and the values are the associated coefficients
    gray_code : bool
        Set to True if to get a gray code ordering of the orders of the Sequential operators
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
    """
    q = QuantumRegister(n,name='q')
    a_qsp = AncillaRegister(n_ancilla_qsp+1,name='a_qsp')
    qc = QuantumCircuit(q,a_qsp)
    
    q_qubits = qc.qubits[:n]
    a_qsp_qubits = qc.qubits[n:]
    
    sequential_operators_list = []
    order_list = list(sequential_info.keys())
    for order in order_list:
        if gray_code:
            sequential_operators_list.append(sequential_operator_qsp(n,order,sequential_info[order],nots=False))
        else:
            sequential_operators_list.append(sequential_operator_qsp(n,order,sequential_info[order]))
    qsp_control_index = 0
    for index,operator in enumerate(sequential_operators_list):
        if gray_code:
            # avoid putting the useless cnots
            if index == 0:
                nots_index = bits_to_zero(int_to_binary(order_list[index],n,reverse=True))
                for j in nots_index:
                    qc.x(q[j])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
            elif index > 0:
                current_index = int_to_binary(order_list[index],n,reverse=True)
                previous_index = int_to_binary(order_list[index-1],n,reverse=True)
                nots_index = changing_index(current_index,previous_index)
                for j in nots_index:
                    qc.x(q[j])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
            if index == len(sequential_operators_list)-1:
                nots_index = bits_to_zero(int_to_binary(order_list[index],n,reverse=True))
                for j in nots_index:
                    qc.x(q[j])
        else:
            qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
        qsp_control_index = (qsp_control_index+1)%(n_ancilla_qsp+1)
    return qc
