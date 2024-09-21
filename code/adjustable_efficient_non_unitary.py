from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
from non_unitary import *

def non_unitary(n,f,d,n_ancilla_diag,n_ancilla_qsp,list_operator_to_implement,decomposition='walsh',gray_code=True,swaps=False):
    """Generates the quantum circuit implementing a non_unitary diagonal operator
    Parameters
    ----------
    n : int
        Number of qubit encoding the position
    f : function
        Function of one variable
    n_ancilla_diag : int
        Number of ancilla qubits available for the parallelization of the diagonal operator
    n_ancilla_qsp : int
        Number of ancilla qubits available for the parallelization of the controlled operation
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
    qA = QuantumRegister(1,name='qA')
    a = AncillaRegister(n_ancilla_diag,name='a')
    a_qsp = AncillaRegister(n_ancilla_qsp,name='a_qsp')
    qc = QuantumCircuit(q,qA,a,a_qsp)

    q_qubits = qc.qubits[:n]
    qA_qubits = qc.qubits[n:n+1]
    a_qubits = qc.qubits[n+1:n+1+n_ancilla_diag]
    a_qsp_qubits = qc.qubits[n+1+n_ancilla_diag:n+1+n_ancilla_diag+n+1+n_ancilla_qsp]

    old_qc_qubits = q_qubits+a_qubits

    if swaps:
        for i in range(int(n/2)):
            qc.swap(q[i],q[n-i-1])
            
    qc.h(qA)

    # copy of qA
    copy_gate_qsp = copy(qA,a_qsp)
    qc.append(copy_gate_qsp,qA_qubits+a_qsp_qubits)

    # copy of main register
    if n_ancilla_diag >= n:
        copy_gate = copy(q,a)
        qc.append(copy_gate,q_qubits+a_qubits)
    
    n_blocks = get_n_blocks(n,n_ancilla_diag)
    n_rotations = len(list_operator_to_implement)
    n_operator_per_block_list = n_operator_per_block(n_rotations,n_blocks)
    n_ancilla_qsp_block = (n_ancilla_qsp+1)//n_blocks
    if decomposition == 'walsh':
        qsp_control_index = 0
        n_op = 2**n #len(list_operator_to_implement)
        walsh_info = walsh_informations_non_unitary(n,n_op,list_operator_to_implement,f,d,gray_code=gray_code)
        #print(walsh_info)
        for i in range(0,n_blocks):
            walsh_info_block = dict(list(walsh_info.items())[:n_operator_per_block_list[i]])
            qc.append(walsh_circuit_non_unitary(n,n_ancilla_qsp,f,walsh_info_block,qsp_control_index,n_ancilla_qsp_block,gray_code=gray_code),old_qc_qubits[i*n:(i+1)*n]+qA_qubits+a_qsp_qubits)
            walsh_info = dict(list(walsh_info.items())[n_operator_per_block_list[i]:])
            qsp_control_index = (qsp_control_index+n_ancilla_qsp_block)%(n_ancilla_qsp+1)
    elif decomposition == 'sequential':
        sequential_info = sequential_informations_non_unitary(n,list_operator_to_implement,f,d,gray_code=gray_code)
        #print(sequential_info)
        for i in range(0,n_blocks):
            sequential_info_block = dict(list(sequential_info.items())[:n_operator_per_block_list[i]])
            qc.append(sequential_circuit_non_unitary(n,n_ancilla_qsp,f,sequential_info_block,gray_code=gray_code),old_qc_qubits[i*n:(i+1)*n]+qA_qubits+a_qsp_qubits)
            sequential_info = dict(list(sequential_info.items())[n_operator_per_block_list[i]:])
    
    # undo copy of main register
    if n_ancilla_diag >= n:
        qc.append(copy_gate.inverse(),q_qubits+a_qubits)

    # undo copy of qA
    qc.append(copy_gate_qsp.inverse(),qA_qubits+a_qsp_qubits)

    qc.h(qA)
    qc.p(-np.pi/2,qA) # NEW
    if swaps:
        for i in range(int(n/2)):
            qc.swap(q[i],q[n-i-1])
            
    return qc
