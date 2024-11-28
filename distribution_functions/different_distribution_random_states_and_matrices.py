import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eigh
import sys 

def Haar_measure_random_state(N: int)-> np.ndarray:
    """
    Generates a random normalized vector of complex numbers.
    Haar measure distributed. 
    
    Parameters:
        N (int): Size of the vector.
        
    Returns:
        np.ndarray: A normalized complex vector |ψ〉.
    """
    # Generate the real and imaginary parts as i.i.d. Gaussian random variables
    real_part = np.random.normal(0, 1, N)
    imag_part = np.random.normal(0, 1, N)
    
    # Combine them to form complex numbers
    X = real_part + 1j * imag_part
    
    # Normalize the vector
    norm = np.linalg.norm(X, ord=2)  # Compute the 2-norm of X
    psi = X / norm  # Normalize
    
    return psi

def Haar_measure_random_state_via_unitary(dim: int) -> np.ndarray:
    """
    Generate a random pure quantum state according to the Haar measure.
    Source: Generating and using truly random quantum states in Mathematica, Jarosław Adam Miszczak
    Scales exponentially with the number of qubits.
    
    Parameters:
    dim (int): Dimension of the Hilbert space (the number of qubits is log2(dim))
    
    Returns:
    np.ndarray: Random quantum state vector of dimension dim.
    """
    # Generate a random unitary matrix from the Haar measure
    U = unitary_group.rvs(dim)
    
    # Select the first column of the unitary matrix
    psi = U[:, 0]

    return psi

# Random quantum states
def random_quantum_state_miszczak(n: int) -> np.ndarray:
    """
    Generate a random pure quantum state according to the Haar measure.
    Source: Generating and using truly random quantum states in Mathematica, Jarosław Adam Miszczak

    
    Parameters:
    n (int): Dimension of the Hilbert space (number of qubits is log2(n))
    
    Returns:
    np.ndarray: Random quantum state vector of dimension n
    """
    # Generate a random unitary matrix from the Haar measure
    U = unitary_group.rvs(n)
    
    # Select the first column of the unitary matrix
    psi = U[:, 0]

    return psi

def ginibre(m: int,n: int)-> np.ndarray:
    """
    m,n >0
    G: random matrix from the Ginibre ensemble.
    Source: Generating and using truly random quantum states in Mathematica, Jarosław Adam Miszczak
    """
    G = np.zeros((m,n),dtype=complex)
    for k in range(0,m):
        for l in range(0,n):
            G[k,l] = np.random.normal()+1j*np.random.normal()
    
    return G


def random_state_HS(n: int,k: int)-> np.ndarray:
    """
    n>=0
    k>=2
    rho: random density matrix distributed according to induced probability measure μn,k 
    (Hilbert Schmidt metric) obtained by tracing out the ancillary system of dimension k.
    Source: Generating and using truly random quantum states in Mathematica, Jarosław Adam Miszczak
    """
    G = ginibre(n,k)
    rho = np.dot(G,np.conj(G.T))
    return rho/np.trace(rho)


def random_state_Bures(n: int)-> np.ndarray:
    """
    n>=0
    rho: andom density matrix distributed according to the probability measure induced by the Bures metric.
    Source: Generating and using truly random quantum states in Mathematica, Jarosław Adam Miszczak
    """
    G = ginibre(n,n)
    U = unitary_group.rvs(n)
    A_1 = np.dot(G,np.conj(G.T))
    A_2 = np.identity(n)+U
    A_3 = np.identity(n)+np.conj(U.T)
    rho = np.dot(A_2,np.dot(A_1,A_3))
    return rho/np.trace(rho)

def random_invertible_matrix(n: int)-> np.ndarray:
    while True:
        # Generate a random n x n matrix
        matrix = np.random.randn(n, n)
        # Check if the matrix is invertible by verifying that its determinant is non-zero
        if np.linalg.det(matrix) != 0:
            return matrix