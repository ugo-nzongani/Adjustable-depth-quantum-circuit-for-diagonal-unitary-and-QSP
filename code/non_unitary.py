from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.library import MCXGate, QFT, CPhaseGate, CRZGate
from walsh import *
from sequential import *
from qsp import *
import torch
from hadamard_transform import hadamard_transform

def reordering(j,n):
    L = dyatic(j/2**n, n)
    jnew = 0
    for i in range(n):
        jnew += L[i]*2**(i)
    return(int(jnew))

def fast_list_walsh_coeff(f,n):
    Values = [f(x/2**n) for x in range(2**n)]    
    L_walsh_coeff_f = np.array(hadamard_transform(torch.Tensor(Values)))/np.sqrt(2**n)
    L_walsh_coeff = [L_walsh_coeff_f[reordering(j,n)] for j in range(2**n)]
    return(L_walsh_coeff)
    
def get_dmax(d,N):
    """Returns the maximal absolute value of function d
    Parameters
    ----------
    d : function
        Function of one variable
    N : int
        Number of point on which d is computed
    Returns
    -------
    int
    """
    maxi = 0
    for i in range(N):
        if abs(d(i/N)) > maxi:
            maxi = abs(d(i/N))
    return maxi

### WALSH ###

def walsh_coeff_non_unitary(j,f,N,d):
    """j-th Walsh coefficient of the N-th Walsh series of f
    Parameters
    ----------
    j : int
        Order of the Walsh coefficient
    f : function
        Function of one variable
    N : int
        Integer representing the number of points on which is computed the Walsh coefficient
    d : function
        Function of one variable
    Returns
    -------
    float
    """
    k = np.array(range(N))/N
    a = 0
    dmax = get_dmax(d,N)
    for i in range(N):
        if callable(f):
            a += f(k[i],d,dmax) * walsh(j,k[i])/N
        else:
            a += f[i] * walsh(j,k[i])/N
    return a

def walsh_informations_non_unitary(n,n_operators,f,d,gray_code=True):
    """Returns a dictionnary whose keys are the order of Walsh operators to implement and the
    values are the associated coefficients
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    n_operators : int
        Number of Walsh operators to implement
    f : function
        Function of one variable
    gray_code : bool
        Set to True if to get a gray code ordering of the orders of the Walsh operators
    d : function
        Function of one variable
    Returns
    -------
    dict
    """
    walsh_dict = {}
    
    # Computation of the Walsh coefficient
    
    dmax = get_dmax(d,2**n)
    def g(x):
        return f(x,d,dmax)
        
    walsh_coeff_list = np.array(fast_list_walsh_coeff(g,n))
    list_operator_to_implement = np.argsort(abs(walsh_coeff_list))[::-1][:n_operators]
    if gray_code:
        gray_list = generate_gray_code(n)
        for i in gray_list:
            if i in list_operator_to_implement:
                walsh_dict[i] = walsh_coeff_list[i] # walsh_coeff_non_unitary(i,f,2**n,d)
    else:         
        for i in range(n_operators):
            walsh_dict[list_operator_to_implement[i]] = walsh_coeff_list[list_operator_to_implement[i]] # walsh_coeff_non_unitary(list_operator_to_implement[i],f,2**n,d)
    return walsh_dict

def walsh_circuit_non_unitary(n,n_ancilla_qsp,f,walsh_info,qsp_control_index,n_ancilla_qsp_block,gray_code=True):
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
        Output of function walsh_informations_non_unitary, it is a dictionnary whose keys are the order of Walsh functions to implement and the
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
    walsh_operators_dagger_list = []
    order_list = list(walsh_info.keys())
    for order in order_list:
        if gray_code:
            walsh_operators_dagger_list.append(walsh_operator_qsp(n,order,-walsh_info[order],cnots=False))
            walsh_operators_list.append(walsh_operator_qsp(n,order,walsh_info[order],cnots=False))
        else:
            walsh_operators_dagger_list.append(walsh_operator_qsp(n,order,-walsh_info[order]))
            walsh_operators_list.append(walsh_operator_qsp(n,order,walsh_info[order]))
    save_qsp_control_index = qsp_control_index
    for index,operator in enumerate(walsh_operators_list):
        if gray_code:
            # avoid putting the useless cnots
            rotation_index = first_one_bit(int_to_binary(order_list[index],n,reverse=False))
            if index == 0:
                cnots_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                for j in cnots_index:
                    qc.cx(q[j],q[rotation_index])
                qc.append(walsh_operators_dagger_list[index],q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
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
                qc.append(walsh_operators_dagger_list[index],q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
            if index == len(walsh_operators_list)-1:
                cnots_index = other_one_bits(int_to_binary(order_list[index],n,reverse=False))
                for j in cnots_index:
                    qc.cx(q[j],q[rotation_index]) 
        else:
            qc.append(walsh_operators_dagger_list[index],q_qubits+[a_qsp_qubits[qsp_control_index]])
            qc.x(a_qsp_qubits[qsp_control_index])
            qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
            qc.x(a_qsp_qubits[qsp_control_index])
        if qsp_control_index + 1 >= n_ancilla_qsp_block:
            qsp_control_index = save_qsp_control_index
        else:
            qsp_control_index = (qsp_control_index+1)%(n_ancilla_qsp+1)
    return qc
    
### SEQUENTIAL ###

def sequential_coeff_non_unitary(j,f,N,d):
    """j-th sequential coefficient
    Parameters
    ----------
    j : int
        Order of the Sequential coefficient
    f : function
        Function of one variable
    N : int
        Integer representing the number of points on which is computed the Sequential coefficient
    d : function
        Function of one variable
    Returns
    -------
    float
    """
    if callable(f):
        dmax = get_dmax(d,N)
        return f(j/N,d,dmax)
    else:
        return f[j]

def sequential_informations_non_unitary(n,list_operator_to_implement,f,d,gray_code=True):
    """Returns a dictionnary whose keys are the order of sequential operators to implement and the
    values are the associated coefficients
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    list_operator_to_implement : int list
        List of orders of the Sequential operators to implement
    f : function
        Function of one variable
    gray_code : bool
        Set to True if to get a gray code ordering of the orders of the sequential operators
    d : function
        Function of one variable
    Returns
    -------
    dict
    """
    sequential = {}
    n_operator_to_implement = len(list_operator_to_implement)
    if gray_code:
        gray_list = generate_gray_code(n)
        for i in gray_list:
            if i in list_operator_to_implement:
                sequential[i] = sequential_coeff_non_unitary(i,f,2**n,d)
    else:          
        for i in range(n_operator_to_implement):
            sequential[list_operator_to_implement[i]] = sequential_coeff_non_unitary(list_operator_to_implement[i],f,2**n,d)
    return sequential

def sequential_circuit_non_unitary(n,n_ancilla_qsp,f,sequential_info,gray_code=True):
    """Generates the quantum circuit implementing the Sequential decomposition of function f with the sequential operators
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
        Output of function sequential_informations_non_unitary, it is a dictionnary whose keys are the order of sequential operator to implement and the
        values are the associated coefficients
    gray_code : bool
        Set to True if to get a gray code ordering of the orders of the sequential operators
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
    sequential_operators_dagger_list = []
    order_list = list(sequential_info.keys())
    for order in order_list:
        if gray_code:
            sequential_operators_dagger_list.append(sequential_operator_qsp(n,order,-sequential_info[order],nots=False))
            sequential_operators_list.append(sequential_operator_qsp(n,order,sequential_info[order],nots=False))
        else:
            sequential_operators_dagger_list.append(sequential_operator_qsp(n,order,-sequential_info[order]))
            sequential_operators_list.append(sequential_operator_qsp(n,order,sequential_info[order]))
    qsp_control_index = 0
    for index,operator in enumerate(sequential_operators_list):      
        if gray_code:
            # avoid putting the useless cnots
            if index == 0:
                nots_index = bits_to_zero(int_to_binary(order_list[index],n,reverse=True))
                for j in nots_index:
                    qc.x(q[j])
                qc.append(sequential_operators_dagger_list[index],q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
            elif index > 0:
                current_index = int_to_binary(order_list[index],n,reverse=True)
                previous_index = int_to_binary(order_list[index-1],n,reverse=True)
                nots_index = changing_index(current_index,previous_index)
                for j in nots_index:
                    qc.x(q[j])
                qc.append(sequential_operators_dagger_list[index],q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
                qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
                qc.x(a_qsp_qubits[qsp_control_index])
            if index == len(sequential_operators_list)-1:
                nots_index = bits_to_zero(int_to_binary(order_list[index],n,reverse=True))
                for j in nots_index:
                    qc.x(q[j])
        else:
            qc.append(sequential_operators_dagger_list[index],q_qubits+[a_qsp_qubits[qsp_control_index]])
            qc.x(a_qsp_qubits[qsp_control_index])
            qc.append(operator,q_qubits+[a_qsp_qubits[qsp_control_index]])
            qc.x(a_qsp_qubits[qsp_control_index])
        qsp_control_index = (qsp_control_index+1)%(n_ancilla_qsp+1)
    return qc
