import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eigh
import sys 

# Properties of a quantum state
def von_neumann_entropy(state: np.ndarray) -> float:
    """
    Compute the von Neumann entropy of a quantum state.
    
    Parameters:
    state (np.ndarray): Quantum state vector or density matrix.
    
    Returns:
    float: Von Neumann entropy of the quantum state.
    """
    # Check if the input is a state vector or a density matrix
    if state.ndim == 1:  # State vector
        rho = np.outer(state, np.conj(state))  # Convert to density matrix
    elif state.ndim == 2 and state.shape[0] == state.shape[1]:  # Density matrix
        rho = state
    else:
        raise ValueError("Input must be a state vector or a density matrix.")
    
    # Compute the eigenvalues of the density matrix
    eigenvalues, _ = eigh(rho)
    
    # Compute the von Neumann entropy
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))  # Adding epsilon to avoid log(0)
    
    return entropy

def partial_trace(rho, dims, keep):
    """
    Compute the partial trace of the density matrix 'rho'.
    
    Parameters:
    rho (np.ndarray): The density matrix.
    dims (list): Dimensions of the subsystems.
    keep (list): Indices of subsystems to keep after tracing out.
    
    Returns:
    np.ndarray: The reduced density matrix.
    """
    dims = np.asarray(dims)
    keep = np.asarray(keep)
    trace_out = np.setdiff1d(np.arange(len(dims)), keep)
    
    # Reshape rho to a tensor
    reshaped_rho = rho.reshape(np.concatenate([dims, dims]))
    
    # Perform the partial trace
    for axis in sorted(trace_out, reverse=True):
        reshaped_rho = np.trace(reshaped_rho, axis1=axis, axis2=axis + len(dims))
    
    # Reshape the result back to a matrix
    dims_keep = dims[keep]
    reduced_rho = reshaped_rho.reshape(np.prod(dims_keep), np.prod(dims_keep))
    
    return reduced_rho

def compute_bond_dimension_truncating(state, dimensions, epsilon=1e-10):
    # Ensure the state is reshaped as a tensor
    state = state.reshape(dimensions)
    
    bond_dims = []
    
    current_state = state
    for i in range(len(dimensions) - 1):
        d1 = np.prod(dimensions[:i+1])
        d2 = np.prod(dimensions[i+1:])
        # Reshape and pad if necessary
        num_elements = current_state.size
        target_size = d1 * d2
        
        if num_elements < target_size:
            # Create a new array with zeros and copy the original data
            padded_state = np.zeros(target_size, dtype=complex)
            padded_state[:num_elements] = current_state.flatten()
            current_state = padded_state.reshape((d1, d2))
        else:
            current_state = current_state.reshape((d1, d2))
        
        # Perform SVD
        # According to np documention, S is an array representing "vector(s) with the singular 
        # values, within each vector sorted in descending order. 
        U, S, V = np.linalg.svd(current_state, full_matrices=False)
        
        # Truncate singular values based on epsilon
        S_truncated = S[S > epsilon]
        
        # Bond dimension is the number of singular values greater than epsilon
        bond_dims.append(len(S_truncated))
        
        # Update current_state for the next partition
        current_state = np.dot(np.diag(S_truncated), V[:S_truncated.size])
    
    # The bond dimension is the maximum bond dimension found
    bond_dimension = max(bond_dims)
    return bond_dimension

def compute_bond_dimension(state, dimensions, epsilon=1e-10):
    # Ensure the state is reshaped as a tensor
    state = state.reshape(dimensions)
    
    bond_dims = []
    
    current_state = state
    for i in range(len(dimensions) - 1):
        d1 = np.prod(dimensions[:i+1])
        d2 = np.prod(dimensions[i+1:])
        
        current_state = current_state.reshape((d1, d2))
        
        # Perform SVD
        # According to np documention, S is an array representing "vector(s) with the singular 
        # values, within each vector sorted in descending order. 
        U, S, V = np.linalg.svd(current_state, full_matrices=False)
        
        # Truncate singular values based on epsilon
        S_truncated = S[S > epsilon]
        
        # Bond dimension is the number of singular values greater than epsilon
        bond_dims.append(len(S_truncated))
    
    # The bond dimension is the maximum bond dimension found
    bond_dimension = max(bond_dims)
    return bond_dimension
# Test and study of the properties of random quantum states

def average_bond_dimension_for_epsilon(num_states, qubit_dim, num_qubits, epsilon_values, truncating=False):
    average_bond_dims = []
    dimensions = [qubit_dim] * num_qubits
    
    for epsilon in epsilon_values:
        bond_dims = []
        for _ in range(num_states):
            # Generate a random quantum state
            state = random_quantum_state(qubit_dim**num_qubits)
            state = state.reshape(dimensions)
            state = state / np.linalg.norm(state)  # Normalize the state
            
            # Compute the bond dimension for the current state
            if truncating:
                bond_dim = compute_bond_dimension_truncating(state, dimensions, epsilon)
            else:    
                bond_dim = compute_bond_dimension(state, dimensions, epsilon)
            bond_dims.append(bond_dim)
        
        # Compute the average bond dimension for the current epsilon
        average_bond_dim = np.mean(bond_dims)
        average_bond_dims.append(average_bond_dim)
    
    return average_bond_dims

def is_density_matrix(rho, tol=1e-10):
    """
    Input:
    rho (np.array): matrix to check if it is a density matrix.
    tol (float): tolerance to the test (numerical error accepted). 
    Output: 
    Bool: True or False according to if rho is a correct density matrix.
    If False, a reason why it is not appear as a message. 
    
    Check if the matrix rho is a density matrix, that is:
    1) rho is Hermitian
    2) trace(rho)==1
    3) rho is positive 
    """
    # Check Hermiticity: rho should equal its conjugate transpose
    if not np.allclose(rho, np.conjugate(rho.T), atol=tol):
        return False, "The matrix is not Hermitian."
    
    # Check that the trace equals 1
    if not np.isclose(np.trace(rho), 1, atol=tol):
        return False, f"The trace of the matrix is not 1 but {np.trace(rho)}."
    
    # Check positivity: all eigenvalues must be non-negative
    eigenvalues = np.linalg.eigvals(rho)
    if not np.all(eigenvalues >= -tol):  # Allowing for numerical tolerance
        return False, "The matrix has negative eigenvalues."
    
    return True, "The matrix is a valid density matrix."


def is_unitary(B, tol=1e-10):
    """    
    Input:
    B (np.array): matrix to check if it is a unitary matrix.
    tol (float): tolerance to the test (numerical error accepted). 
    
    Output: 
    Bool: True or False according to if B is unitary. 
    """
    # Calculate the conjugate transpose of B
    B_dagger = np.conjugate(B.T)
    
    # Check if B_dagger * B is close to the identity matrix
    identity_matrix = np.eye(B.shape[0])  # Identity matrix of same size as B
    result = np.dot(B_dagger, B)
    
    # Check if the result is approximately the identity matrix
    return np.allclose(result, identity_matrix, atol=tol)


def S_mn(m,n):
    """ 
    If a quantum system of Hilbert space dimension mn is in a random pure state, the average entropy 
    of a subsystem of dimension m ≤ n known and given by the output of this function. 
    
    Inputs:
    m: dimension of the subsystem A.
    n: dimension of the subsystem B. 
    
    Return:
    Theoretical average entropy. 
    
    Reference:
    Don N. Page. Average entropy of a subsystem. Physical Review Letters, 71(9):1291–1294, August 1993
    """
    if m>n:
        assert("m must be smaller or equal than n.")
    result = sum([1/k for k in range(n+1,m*n+1,1)])
    return result -(m-1)/(2*n)