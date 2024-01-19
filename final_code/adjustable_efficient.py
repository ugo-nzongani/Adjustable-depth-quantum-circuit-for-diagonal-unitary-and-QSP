from walsh import *
from sequential import *

def general_framework(n,f,n_ancilla,list_operator_to_implement,decomposition='walsh',gray_code=True,swaps=False):
    """Generates the quantum circuit implementing the sequential decomposition of function f with the sequential operators
        whose order are contain in sequential_info 
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    f : function
        Function of one variable
    n_ancilla : int
        Number of ancilla qubits available
    list_operator_to_implement : int list
        List containing the indices of the operators to implement, an exact implementation requires to
        implement all of them
    decomposition : str
        'walsh' or 'sequential', it indicates the type of the decomposition for the circuit
    gray_code : bool
        Set to True if to get a gray code ordering of the operators
    swaps : bool
        Set to True to inverse the bit order of the main register
    Returns
    -------
    qiskit.circuit.quantumcircuit.QuantumCircuit
    """
    q = QuantumRegister(n,name='q')
    a = AncillaRegister(n_ancilla,name='a')
    qc = QuantumCircuit(q,a)

    if swaps:
        for i in range(int(n/2)):
            qc.swap(q[i],q[n-i-1])
    
    if n_ancilla >= n:
        copy_gate = copy(q,a)
        qc.append(copy_gate,qc.qubits)

    n_blocks = get_n_blocks(n,n_ancilla)
    n_rotations = len(list_operator_to_implement)
    n_operator_per_block_list = n_operator_per_block(n_rotations,n_blocks)

    if decomposition == 'walsh':
        walsh_info = walsh_informations(n,list_operator_to_implement,f,gray_code=gray_code)
        for i in range(0,n_blocks):
            walsh_info_block = dict(list(walsh_info.items())[:n_operator_per_block_list[i]])
            qc.append(walsh_circuit(n,f,walsh_info_block,gray_code=gray_code),qc.qubits[i*n:(i+1)*n])
            walsh_info = dict(list(walsh_info.items())[n_operator_per_block_list[i]:])

    elif decomposition == 'sequential':
        sequential_info = sequential_informations(n,list_operator_to_implement,f,gray_code=gray_code)
        for i in range(0,n_blocks):
            sequential_info_block = dict(list(sequential_info.items())[:n_operator_per_block_list[i]])
            qc.append(sequential_circuit(n,f,sequential_info_block,gray_code=gray_code),qc.qubits[i*n:(i+1)*n])
            sequential_info = dict(list(sequential_info.items())[n_operator_per_block_list[i]:])

    if n_ancilla >= n:
        qc.append(copy_gate.inverse(),qc.qubits)
        
    if swaps:
        for i in range(int(n/2)):
            qc.swap(q[i],q[n-i-1])
            
    return qc
