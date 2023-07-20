# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:16:07 2023

@author: jzylberman

Quantum State Preparation with Walsh Series
"""



import numpy as np
from bitstring import BitArray
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.transforms import Bbox

from tqdm import tqdm
 

# Import Qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import qiskit.quantum_info as qi
from qiskit import Aer, transpile

"""function test"""

def gaussian1(x):
    #sigma=0.1
    sigma=1
    mu=0.5
    return(np.exp(-(x-mu)**2/(2*sigma**2))/(sigma))

def gaussian2(x):
    #sigma=0.1
    sigma=0.1
    mu=0.5
    return(np.exp(-(x-mu)**2/(2*sigma**2))/(sigma))

def bimodalgaussian(x):
    s=0.1
    mu1=0.25
    sigma1=0.3
    mu2=0.75
    sigma2=0.04
    return((1-s)*np.exp(-(x-mu1)**2/(2*sigma1**2))/(sigma1)+s*np.exp(-(x-mu2)**2/(2*sigma2**2))/(sigma2))
    

def lorentzian(x):
    #Gamma=0.05
    #Gamma=0.1
    Gamma=1
    mu=0.5
    return((1/(Gamma))/(1+((x-mu)/(Gamma/2))**2))

def log_normal(x):
    sigma=0.25/5
    mu=0
    a=3
    b=0.001
    return(np.exp(-(np.log(a*x+b)-mu)**2/(2*sigma**2))/((a*x+b)*sigma))


def inverse2(x):
    return(1/(1-x))


def val(x):
    return(abs(x-0.25)-abs(x-0.5)+abs(x-0.75))

def f5(x):
    return(np.exp(-(x-0.5)**2))
def gauss(x):
    return(np.exp(-100*(x-0.5)*(x-0.5))+1)


def sinc(x):
    return(np.sinc(np.pi*6*(x-0.5)))

def s(x):
    return(np.sin(2*np.pi*(x)))

def s2(x):
    return(np.sin(4*np.pi*x)+np.sin(2*np.pi*x))

def c(x):
    return(np.cos(2*np.pi*x))

def sq(x):
    return(np.sqrt(abs(x-0.25)))

def test(x):
    return(gauss(x)*np.exp(1j*np.pi*s(x)/2))

def discontinuous(x):
    return(np.sin(2*np.pi*(x-1/3)*Walsh(2,x)))

"""Walsh functions"""

def Walsh(j,x): 
    """
    Parameters
    ----------
    j : int
        Order of the Walsh function.
    x : float
        real number in [0,1].

    Returns
    -------
    The value of j-th Walsh function at position x.
    """
    jbin=bin(j)
    lj=len(jbin)-2
    
    X=dyatic(x,lj)

    p=0
    for i in range(lj):
        p+=int(int(jbin[-1-i])*X[i])
    return((-1)**p)

def dyatic(x,n):
    """
    Parameters
    ----------
    x : float
        real number in [0,1].
    n : int
        index for the truncation of the dyatic expansion of x.

    Returns
    -------
    return the list of coefficient of the dyatic expansion of x up to order n

    """
    L=np.zeros((n))
    a=x
    for i in range(n):
        if a-1/2**(i+1)>=0:
            L[i]=1
            a=a-1/2**(i+1)
    return(L)

def Walsh_coeff(j,f,N):
    """
    Parameters
    ----------
    j : int
        order of the Walsh coefficient.
    f : function
        Function of one variable.
    N : int
        integer representing the number of points on which is computed the Walsh coefficient.
    Returns
    -------
    j-th Walsh coeffient of the N-th Walsh series of f.

    """
    K=np.array(range(N))/N
    
    a=0
    for i in range(N):
        a+=f(K[i])*Walsh(j,K[i])/N
    return(a)


def List_walsh_coeff(f,N):
    L=[]
    for j in range(N):
        L.append(Walsh_coeff(j,f,N))
    return(L)


def generate_gray_list(my_val):
    """
    Parameters
    ----------
    my_val : int

    Returns
    -------
    List of the first 2**my_val binary numbers in an order such that only one digit differs from one number to the next one.

    """
    if (my_val <= 0):
       return
    my_list = list()
    my_list.append("0")
    my_list.append("1")
    i = 2
    j = 0
    while(True):
       if i >= 1 << my_val:
          break
       for j in range(i - 1, -1, -1):
          my_list.append(my_list[j])
       for j in range(i):
          my_list[j] = "0" + my_list[j]
       for j in range(i, 2 * i):
          my_list[j] = "1" + my_list[j]
       i = i << 1
       
    #for i in range(len(my_list)):
    #    print(my_list[i])
    return(my_list)

""" quantum circuits and operators"""

def walsh_operator(j,theta,n,threshold=0): #n number of qubits, a walsh coefficient, X is the register
    """
    Parameters
    ----------
    j : int
        order of the Walsh function.
    theta : float
        angle of the rotation, usually corresponding to the j-Walsh coefficient times a constant.
    n : int
        number of qubits encoding the position axis.

    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D j-th Walsh operators

    """
    q = QuantumRegister(n,'q')
    qc= QuantumCircuit(q)
    
    if abs(theta)<=threshold:
        return(qc.to_gate(label="walsh1D"))
    #print(theta,j)
    
    lj=bin(j)
    nj=len(lj)-3
    
    for i in range(nj):
        if lj[-i-1]=='1':
            qc.cx(i,nj)
    if j==0:
        qc.x(0)
        qc.p(theta,0)
        qc.x(0)
        qc.p(theta,0)
    else:
        qc.rz(-2*theta,nj)
    
    for i in range(nj):
        if lj[-i-1]=='1':
            qc.cx(i,nj)
    walsh1D = qc.to_gate(label="walsh1D")
    #print(j,theta,qc.draw())
    return(walsh1D)
            

def norm(f,N):
    a=0
    K=np.array(range(N))/N
    for i in range(N):
        a+=f(K[i])*np.conjugate(f(K[i]))
    return(np.real(np.sqrt(a)))    
           
def walsh_graycode_qc(G,L,m,n,threshold=0): 
    """ 
    Parameters
    ----------
    G : List
        Gray_list of the 2**n first binary numbers.
    L : array
        Array of 2**mx*2**my Walsh coeffients/angles
    m : int
        number of qubits on the position register on which is encoding the diagonal unitary.
    nx : int
        number of qubits encoding the x-axis.
    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D diagonal unitary associated to L

    """  
    q = QuantumRegister(n,'q')
    qc= QuantumCircuit(q)
    
    for j in range(2**m):
        j1=BitArray(bin=G[j]).uint
        walsh1D=walsh_operator(j1,L[j1],n,threshold)
        qubits = [i for i in q]
        qc.append(walsh1D,qubits)  
    
    diagonal_unitary_1D = qc.to_gate(label="diagonal_unitary_1D")
    return(diagonal_unitary_1D)


         
def walsh_decreasing_order_qc(L,m,n,threshold=0,n_operators=0): 
    """ 
    Parameters
    ----------
    L : array
        Array of 2**mx*2**my Walsh coeffients/angles
    m : int
        number of qubits on the position register on which is encoding the diagonal unitary.
    nx : int
        number of qubits encoding the x-axis.
    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D diagonal unitary associated to L

    """  
    q = QuantumRegister(n,'q')
    qc= QuantumCircuit(q)
    I=np.argsort(abs(np.array(L)))
    for j in range(2**m):
        j1=I[-j-1]
        if abs(L[j1])<threshold:
            break
        if j>n_operators:
            break
        walsh1D=walsh_operator(j1,L[j1],n,threshold)
        qubits = [i for i in q]
        qc.append(walsh1D,qubits)  
    
    diagonal_unitary_1D = qc.to_gate(label="diagonal_unitary_1D")
    return(diagonal_unitary_1D)
        


def diagonal_unitary_1D(f,m,n,threshold=0,n_operators=0,method='graycode',swap_option=False):
    """
    Parameters
    ----------
    f : function
        target real function defined on [0,1].
    m : int
        number of qubits on which is implemented the diagonal unitary.
    n : int
        number of qubits.
    swap_option : Boolean, optional
        If Yes, perform the swapping of the quantum register. The default is False.

    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D diagonal unitary associated to f
    """
    q = QuantumRegister(n,'q')
    qc = QuantumCircuit(q)
    
    
    L=List_walsh_coeff(f,2**m)
    
    if swap_option:
        for i in range(int((n)/2)):
            qc.swap(i,n-i-1)
    
    if method=='graycode':
        G=generate_gray_list(m)
        fullwalsh=walsh_graycode_qc(G,L,m,n,threshold)
    elif method=='decreasing_order':
        fullwalsh=walsh_decreasing_order_qc(L,m,n,threshold,n_operators)
        
    qubits = [i for i in q]
    qc.append(fullwalsh,qubits)  
    
    if swap_option:
        for i in range(int((n)/2)):
            qc.swap(i,n-i-1)
            
            
    #simulator = Aer.get_backend('qasm_simulator')
    #qc_trans = transpile(qc, simulator)
    #circ_trans = transpile(qc, basis_gates=['u','cx','cp','cz'])
    #print("size_brut:", circuit.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",circ_trans2.size())
    #print("depth_brut:", circuit.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",circ_trans2.depth())
    
    diagonal_unitary = qc.to_gate(label="diagonal_unitary")
    return(diagonal_unitary) 


def plot1D_verification(f,m,n,threshold=0,n_operators=100,method='graycode',swap_option=False):
    
    q = QuantumRegister(n,'q')
    
    circuit = QuantumCircuit(q)
    qubits = [i for i in q]
    
    for i in range(n): #Hadamard tower to initialize
        circuit.h(i)
        
    circuit.append(diagonal_unitary_1D(f,m,n,threshold,n_operators,method,swap_option),qubits)    
    
    
    stv = qi.Statevector.from_instruction(circuit)    
    
    wavevector=stv.data
    #print(circuit.draw(),wavevector)
    l=len(wavevector)
    
    X=np.array(range(2**n))/2**n
    
    norm1=0 
    for i in range(l):
        norm1+=np.conjugate(wavevector[i])*wavevector[i]
    norm1=np.sqrt(norm1)
    phase=np.angle(wavevector)
    wavevector=wavevector/norm1
    

    plt.plot(X,phase,c='r',marker='x',label="quantum state |ψ(x)|")
    #plt.plot(KN1,g2(KN1)/2**((n-m)/2),c='b',label="Target function f(x)")
    plt.plot(X,f(X),c='b',label="Target function f(x)")
    plt.title("m="+str(m)+", n="+str(n))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("")
    plt.legend()  
    plt.show()
    
    
""" Quantum State Preparation """

def fidelity(psi1,psi2,n):
    F=0+0j
    for i in range(2**n):
        F+=np.conj(psi1[i])*psi2[i]
    return(abs(F)**2)
 
    
            
def qc_initial_state_preparation(f,m,n,threshold=0,n_operators=0,method='graycode',swap_option=False):
    
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a') #ancilla qubit
    
    circuit = QuantumCircuit(q,a)
    
    if swap_option:
        for i in range(int((n)/2)):
            circuit.swap(i,n-i-1)
    
    for i in range(n): #Hadamard tower to initialize
        circuit.h(q[i])
    circuit.h(a[0])
        
    diagonal_unitary=diagonal_unitary_1D(f,m,n,threshold,n_operators,method,swap_option=False)
    
    controlled_diagonal_unitary = diagonal_unitary.control()
    
    qubits = [a[0]]+[i for i in q]
    circuit.append(controlled_diagonal_unitary,qubits)
    
    circuit.h(a[0])
    circuit.p(-np.pi/2,a[0])
    
    if swap_option:
        for i in range(int((n)/2)):
            circuit.swap(i,n-i-1)
    
    simulator = Aer.get_backend('qasm_simulator')
    circuit_trans = transpile(circuit, simulator)
    #circ_trans2 = transpile(circuit, basis_gates=['u','cx','cp','cz'])
    #print("size_brut:", circuit.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",circ_trans2.size())
    #print("depth_brut:", circuit.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",circ_trans2.depth())
    
    #print(circuit.draw())
    QSP_1D = circuit_trans.to_gate(label="QSP_1D")
    return(QSP_1D) 


    
def plot_QSP_1D(f,m,n,eps0,threshold=0,n_operators=10000,method='graycode',swap_option=False,plot_option=True):
    
    def g(x):
        return(-f(x)*eps0)
    
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a') #ancilla qubit
    #c = ClassicalRegister(1,'c')
    circuit = QuantumCircuit(q,a)
    
    qubits = [i for i in q]+[a[0]]
    
    QSP_1D=qc_initial_state_preparation(g,m,n,threshold,n_operators,method,swap_option)
    
    circuit.append(QSP_1D,qubits)
    #circuit.measure(a[0],c[0])
    #print(circuit.draw())
    
    simulator = Aer.get_backend('statevector_simulator')
    #simulator = Aer.get_backend('qasm_simulator')
    circ_trans = transpile(circuit, simulator, optimization_level=3)
    circ_trans2 = transpile(circuit, basis_gates=['u','cx','cp','cz'],optimization_level=0)
    #print("size_brut:", circuit.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",circ_trans2.size())
    #print("depth_brut:", circuit.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",circ_trans2.depth())
    print("m="+str(m),dict(circ_trans.count_ops()))
    
    
    stv= qi.Statevector.from_instruction(circuit)
    
    L=stv.data
    l=len(L)
    #print(L)
    X=np.array(range(2**n))/2**n
    
    """ We take values of the vector for which the ancilla qubit is in state 1 and we need to renormalize."""
    renorm=0 
    for i in range(int(l/2)):
        renorm+=np.conjugate(L[i+int(l/2)])*L[i+int(l/2)]
    renorm1=np.sqrt(renorm)
    
    Wavevector=L[int(l/2):l]/renorm1
    
    #Wavevector=np.array([L[2*i] for i in range(int(l/2))])/renorm1
    
    implemented_state=np.real(np.array(Wavevector))
    target_state=f(X)/norm(f,2**n)
    
    F1=fidelity(target_state,Wavevector,n)
    #print(1-F1)
    #print(A)
    #print((L[int(l/2):l]/norm1))
    if plot_option==True:
        plt.plot(X,target_state,c='b',ls='-',marker='',label="Target state",linewidth=2.5)
       
        #plt.plot(KN1,g2(KN1)/2**((n-m)/2),c='b',label="Target function f(x)")
        #F2=plot_truncated_walsh_serieM_final(g,n,m)
        
        
        plt.plot(X,implemented_state,c='r',ls='-',marker='',label='Implemented state')
        
        plt.title('$1$'+"D Quantum State Preparation")
        matplotlib.rc('font',family='serif',size=11)
        
        plt.grid()
        plt.xlabel('$x$')
        plt.xticks([0,1/4,1/2,3/4,1],["0","0.25","0.5","0.75","1"])
        plt.legend()
        plt.ylabel('$f_1(x)$')
        bbox = Bbox([[-0.2, 0], [6.2, 4]])
        plt.savefig('QSP of bimodalgaussian with n='+str(n)+', m='+str(m)+ 'eps0='+str(eps0),dpi='figure',format='pdf',bbox_inches=bbox)
        plt.show()
    #plt.legend()
    return(1-F1,circ_trans2.depth())
    