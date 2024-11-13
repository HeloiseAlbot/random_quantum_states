import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import sys 
# Random quantum states
def random_quantum_state(n: int) -> np.ndarray:
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
def ginibre(m,n):
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

def random_state_HS(n,k):
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

def random_state_Bures(n):
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


# Parameters
# num_states = 1000  # Number of random quantum states
# qubit_dim = 2      # Dimension of each qubit (2 for qubits)
# num_qubits = 10   # Number of qubits
# epsilon_values = np.logspace(-12, 0, 50)  # Range of epsilon values, logarithmically spaced

# # Compute the average bond dimension for each epsilon
# average_bond_dims = average_bond_dimension_for_epsilon(num_states, qubit_dim, num_qubits, epsilon_values)
# average_bond_dims_truncating = average_bond_dimension_for_epsilon(num_states, qubit_dim, num_qubits, epsilon_values, truncating=True)
# np.save(f'average_bond_dims_nsamples{num_states}_nqubits{num_qubits}.npy', average_bond_dims)
# np.save(f'average_bond_dims_truncating_nsamples{num_states}_nqubits{num_qubits}.npy', average_bond_dims_truncating)
# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(epsilon_values, average_bond_dims, marker='o', linestyle='-', color='b', label='without truncation')
# plt.plot(epsilon_values, average_bond_dims_truncating, marker='o', linestyle='-', color='r', label='truncating')
# plt.xscale('log')
# plt.xlabel('Truncation Threshold (epsilon)')
# plt.ylabel('Average Bond Dimension')
# plt.title('Average Bond Dimension vs. Truncation Threshold')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig(f'average_bd_vs_truncation_threshold_nsamples{num_states}_nqubits{num_qubits}.png')
# plt.show(block=False)

# Parameters
# sys.argv = [name_of_the_program, epsilon]

n_qubits  = 10
n_qubits_subsystem = 5
qubit_dim = 2
dim = qubit_dim**n_qubits  # Total dimension of the system 
dimensions = [qubit_dim] * n_qubits
entropy_bins = 20   # Number of bins for the entropy axis
subsys_dim = qubit_dim**n_qubits_subsystem  # Dimension of the subsystem 
num_states = 10000000 # Number of random states to generate
num_bins = 20  # Number of bins for the histogram
# python3 random_state.py $epsilon $simulation
epsilon = float(sys.argv[1])
simulation = int(sys.argv[2])
# Initialize lists to hold entropy and bond dimension values
entropy_values = []
bond_dimension_values = []
# Generate random states and compute entropy and bond dimension
for _ in range(num_states):
    state = random_quantum_state(dim)  # Simulate a random quantum state
    entropy_value = von_neumann_entropy(state)  # Compute entropy
    bond_dim_value = compute_bond_dimension_truncating(state, dimensions, epsilon=epsilon)  # Compute bond dimension
    entropy_values.append(entropy_value)
    bond_dimension_values.append(bond_dim_value)
    np.save(f'Histogram_entropy_vs_bd_entropy_nsamples{num_states}_nqubits{n_qubits}_epsilon{epsilon}_sim{simulation}.npy', entropy_values)
    np.save(f'Histogram_entropy_vs_bd_bond_dimension_values_nsamples{num_states}_nqubits{n_qubits}_epsilon{epsilon}_sim{simulation}.npy', bond_dimension_values)

# Convert lists to numpy arrays for easier processing
entropy_values = np.array(entropy_values)
bond_dimension_values = np.array(bond_dimension_values)

# Define bins for entropy and bond dimension
entropy_bin_edges = np.linspace(entropy_values.min(), entropy_values.max(), entropy_bins + 1)
bond_dim_bin_edges = np.arange(min(bond_dimension_values)-1, max(bond_dimension_values) + 1, 1)
# Create a 2D histogram (entropy vs bond dimension)
hist, xedges, yedges = np.histogram2d(entropy_values, bond_dimension_values, bins=[entropy_bin_edges, bond_dim_bin_edges])
# Compute the probability by normalizing the histogram
hist = hist / np.sum(hist)
# Plot the 3D histogram
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# Create the meshgrid for the axes
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1] - xedges[0]),
                         yedges[:-1] + 0.5 * (yedges[1] - yedges[0]), indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
# The heights of the bars
dx = dy = (xedges[1] - xedges[0]) * np.ones_like(zpos)
dz = hist.ravel()
# Plot the bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
# Set labels
ax.set_xlabel('Entropy')
ax.set_ylabel('Bond Dimension')
ax.set_zlabel('Probability')
ax.set_title('3D Histogram of Entropy vs Bond Dimension')
plt.savefig(f'histogram_entropy_vs_bd_nsamples{num_states}_nqubits{n_qubits}_epsilon{epsilon}.png')
plt.show()
