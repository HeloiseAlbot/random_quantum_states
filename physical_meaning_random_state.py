#%%
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.optimize import root
from scipy.optimize import minimize

from scipy.linalg import block_diag, eigh
from scipy.stats import unitary_group
from itertools import product
from scipy.stats import norm

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

#%% Generate random state defined by the level of entanglement between the lattice sites
def random_unitary_matrix(n):
    """
    Generate a random complex matrix of size n x n.
    Only taking the Q from the QR decomposition is wrong according to (page 7): 
    How to generate random matrices from the classical compact groups
    https://arxiv.org/abs/math-ph/0609050.
    """
    real_part = np.random.rand(n, n)
    imag_part = np.random.rand(n, n)
    z = (real_part + 1j * imag_part)/(2**0.5)
    # Use QR decomposition to obtain a unitary matrix
    Q, R = np.linalg.qr(z)
    d = np.diag(R)
    Q = Q * (d / np.abs(d)) 
    return Q 

def reshape_array_before_middle(A, d, l):
    """
    Reshape A of dimension (d^l, d^l) into B of dimension (d, d^(l-1), d^l)
    """
    d_l = int(d ** l)
    d_l_minus_1 = int(d ** (l - 1))
    # Initialize B with the desired shape
    B = np.zeros((d, d_l_minus_1, d_l), dtype=A.dtype)
    # Fill B by taking slices from A
    for i in range(d):
        B[i, :, :] = A[i * d_l_minus_1: (i + 1) * d_l_minus_1, :]
    return B

def reshape_array_after_middle(A, d, l):
    """
    Reshape A of dimension (d^dimension, d^dimension) into B of dimension (d, d^(L-l+1), d^(L-l))
    dimension = d**(L-l+1)
    """
    dimension = A.shape[0] 
    d_L_minus_l = int(dimension/d)
    # Initialize B with the desired shape
    B = np.zeros((d, dimension, d_L_minus_l), dtype=A.dtype)
    # Fill B by taking slices from A
    for i in range(d):
        B[i, :, :] = A[:, i * d_L_minus_l: (i + 1) * d_L_minus_l]
    return B


def block_diagonal_from_rho(rho, d):
    """
    Create a block diagonal matrix with d blocks, each equal to the matrix rho.
    
    Parameters:
    rho (np.ndarray): The matrix to repeat in the block diagonal.
    d (int): The number of blocks. Here, it corresponds to the dimension of a single site, the dimension 
    of a qudit, in the case of qubit d=2.
    
    Returns:
    block_matrix (np.ndarray): The block diagonal matrix with d blocks of rho.
    """
    # Create a list of d copies of rho
    block_list = [rho] * d
    # Return the block diagonal matrix using scipy's block_diag
    return block_diag(*block_list)


def f_from_M_before_middle(M, rho_l, d, l):
    """
    M is of dimension (d^{n},d^{n})).
    rho_l is of diemnsion (d^{n-1},d^{n-1}).
    Should return a matrix of shape (d^n, d^n).
    """
    d_l = int(d**l)
    M = M.reshape((d_l, d_l))
    M_tensor = reshape_array_before_middle(M, d, l) 
    sum_product = np.zeros((int(d**(l-1)), int(d**(l-1))), dtype=np.complex128)
    for sigma in range(d):
        M_sigma = M_tensor[sigma, :, :]
        sum_product += M_sigma @ rho_l @ M_sigma.conj().T
    return sum_product

# Define the loss function to minimize |f(M) - rho_2|^2 (Frobenius norm)
def loss_function_M_before_middle(M, rho_l, rho_l_minus_one, d, l):
    fM = f_from_M_before_middle(M, rho_l, d, l)
    return np.linalg.norm(fM - rho_l_minus_one, 'fro')**2

# Constraint function: sum_sigma M[sigma,:,:].H @ M[sigma,:,:] = Identity
def constraint_M_before_middle(M, d, l):
    M = M.reshape(d**l, d**l)
    dimension_M = M.shape[0]

    identity = np.eye(dimension_M, dtype=np.complex128)
    M_tensor = reshape_array_before_middle(M, d, l)
    sum_product = np.zeros((int(dimension_M), int(dimension_M)), dtype=np.complex128)    
    for sigma in range(d):
        M_sigma = M_tensor[sigma, :, :]
        sum_product += M_sigma.conj().T @ M_sigma
    
    return np.linalg.norm(sum_product - identity, 'fro')  # Should equal 0 for valid M

# Constraint function: sum_sigma M[sigma,:,:].H @ M[sigma,:,:] = Identity
def constraint_M_after_middle(M, d, l, L):
    dimension = d**(L-l+1)
    M = M.reshape(dimension, dimension)
    dimension_M = M.shape[0]

    identity = np.eye(int(dimension_M/d))
    M_tensor = reshape_array_after_middle(M, d, l)
    sum_product = np.zeros((int(dimension_M/d), int(dimension_M/d)), dtype=np.complex128)
    
    for sigma in range(d):
        M_sigma = M_tensor[sigma, :, :]
        sum_product += M_sigma.conj().T @ M_sigma
    return np.linalg.norm(sum_product - identity, 'fro')  # Should equal 0 for valid M

def f_from_M_after_middle(M, rho_l, d, l, L):
    """
    M is of dimension (d^{n},d^{n})).
    rho_l is of diemnsion (d^{n-1},d^{n-1}).
    Should return a matrix of shape (d^n, d^n).
    """
    dimension = d**(L-l+1)
    M = M.reshape(dimension, dimension)
    R = block_diagonal_from_rho(rho_l, d)
    return M @ R @ np.conj(M.T)

def loss_function_M_after_middle(M, rho_l, rho_l_minus_one, d, l, L):
    fM = f_from_M_after_middle(M, rho_l, d, l, L)
    return np.linalg.norm(fM - rho_l_minus_one, 'fro')**2

#%% Produce random state from the mps construction. 
def produce_a_random_state(L, d=2):
    """
    Input:
    L (int): number of site (assumed even for simplicity for the moment).
    d (int): dimension of the hilbert space of a single site (for qubit d=2).
    
    Output:
    list_M (list): list of the tensors M_optimized ordered increasingly. 
    list_objective_error
    list_left_normalization_error
    rho_middle
    
    The function constructs in the decreasing order, the matrix M to produce a random product state. 
    """
    rho_l = np.array([[1]]) # Initial density matrix rho_L.
    list_M = [] # I store the matrices M in a list. 
    list_objective_error = []
    list_left_normalization_error = []
    for l in range(L, int(L/2), -1): # l> L/2
        d_L_minus_l_plus_1 = int(d**(L-l+1))
        # Random unitary matrix (rho is a density matrix so Hermitian so diagonalizable by a unitary matrix).
        Q = random_unitary_matrix(d_L_minus_l_plus_1) 
        D = np.diag(random_eigenvalues_from_n_sphere(d_L_minus_l_plus_1)) # Uniform distribution, to be modified later. 
        rho_l_minus_one = Q @ D @ np.conj(Q).T
        # Initialize M randomly 
        M_initial = np.random.randn(d_L_minus_l_plus_1, d_L_minus_l_plus_1) + 1j * np.random.randn(d_L_minus_l_plus_1, d_L_minus_l_plus_1)
        # Constraints dictionary for scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint_M_after_middle, 'args': (d, l, L)}
        # Optimize M to minimize the loss while satisfying the constraint
        result = minimize(loss_function_M_after_middle, M_initial, args=(rho_l, rho_l_minus_one, d, l, L), 
                          constraints=constraints, method='SLSQP', options={'disp': False})
        # Put options={'disp': True} to have the information about the optimization. 
        # Reshape the result back to the original shape
        M_optimized = result.x.reshape(d_L_minus_l_plus_1, d_L_minus_l_plus_1)
        list_M.insert(0, reshape_array_after_middle(M_optimized, d, l))
        list_objective_error.insert(0, check_objective_after_middle(M_optimized, rho_l, rho_l_minus_one, d, l, L))
        list_left_normalization_error.insert(0, check_left_normalization_after_middle(M_optimized, d, l, L))
        rho_l = rho_l_minus_one
    rho_middle = rho_l
    for l in range(int(L/2), 0, -1): #l <= L/2
        d_l = int(d**(l))
        d_l_minus_one = int(d**(l-1))
        # Random unitary matrix (rho is a density matrix so Hermitian so diagonalizable by a unitary matrix).
        Q = random_unitary_matrix(d_l_minus_one) 
        D = np.diag(random_eigenvalues_from_n_sphere(d_l_minus_one)) # Uniform distribution, to be modified later. 
        rho_l_minus_one = Q @ D @ np.conj(Q).T
        # Find M minimising our objective function with the left normalization constraint. 
        M_initial = np.random.randn(d_l, d_l) + 1j * np.random.randn(d_l, d_l)
        # Constraints dictionary for scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint_M_before_middle, 'args': (d, l)}
        # Optimize M to minimize the loss while satisfying the constraint
        result = minimize(loss_function_M_before_middle, M_initial, args=(rho_l, rho_l_minus_one, d, l), 
                          constraints=constraints, method='SLSQP', options={'disp': False})
        # Reshape the result back to the original shape
        M_optimized = result.x.reshape(d_l, d_l)
        list_M.insert(0, reshape_array_before_middle(M_optimized, d, l))
        list_objective_error.insert(0, check_objective_before_middle(M_optimized, rho_l, rho_l_minus_one, d, l))
        list_left_normalization_error.insert(0, check_left_normalization(M_optimized, d, l))
        rho_l = rho_l_minus_one
    return list_M, list_objective_error, list_left_normalization_error, rho_middle


def vector_state_from_mps(list_M: list, L:int, d=2)->np.array:
    """
    Input:
    list_M (list): list of the matrices M_optimized ordered increasingly. The matrices M_optimized[l] for l
    in [1, L/2] contained the matrices A_l^{\sigma_l} are stored horizontally. The matrices M_optimized[l] for l
    in [L/2+1, L] contained the matrices A_l^{\sigma_l} are stored vertically.
    L (int): the number of lattice sites. 
    d (int): The size of the sigma_l index, the dimension of the qudit.
    
    Output:
    vector_state (np.array): vector_state[sigma_1, sigma_2, ..., sigma_L] holds the desired matrix product
    """
    # Initialize vector_state with shape (d, d, ..., d) repeated L times
    vector_state_shape = (d,) * L
    vector_state = np.zeros(vector_state_shape, dtype=object)
    # Iterate over all combinations of sigma_1, sigma_2, ..., sigma_L
    for sigma_tuple in product(range(d), repeat=L):
        # Initialize product as identity matrix of size (n x n)
        product_matrix = np.eye(1, dtype="complex")

        # Multiply A_1^{sigma_1} @ A_2^{sigma_2} @ ... @ A_L^{sigma_L}
        for l, sigma_l in enumerate(sigma_tuple):
            A = list_M[l]
            product_matrix = product_matrix @ A[sigma_l, :, :]

        # Assign the result to vector_state at the index sigma_1, ..., sigma_L
        vector_state[sigma_tuple] = product_matrix

    # Now vector_state[sigma_1, sigma_2, ..., sigma_L] holds the desired matrix product

    return vector_state

def random_point_on_the_n_sphere(n, r=1):
    # Step 1: Generate the random angles
    angles = np.zeros(n-1)
    angles[:-1] = np.random.uniform(0, np.pi, n - 2)   # φ1, φ2, ..., φ_{n-2} in [0, π]
    angles[-1] = np.random.uniform(0, 2 * np.pi)       # φ_{n-1} in [0, 2π)
    # Step 2: Compute x1, x2, ..., xn
    coordinates = np.zeros(n)n
    product_sin = 1  # Initialize the cumulative product of sin terms
    for i in range(n - 1):
        if i > 0:
            product_sin *= np.sin(angles[i - 1])
        coordinates[i] = r * product_sin * np.cos(angles[i])
    # Last coordinate xn with full product of sines
    coordinates[-1] = r * product_sin * np.sin(angles[-1])
    
    return coordinates

def random_eigenvalues_from_n_sphere(n, r=1):
    if n ==1:
        return np.array([[1]])
    else:
        coordinates = random_point_on_the_n_sphere(n, r)
        return np.sort(np.square(coordinates))[::-1]


def vector_state_from_mps(L, d=2):
    _, _, _, rho_middle = produce_a_random_state(L, d)
    # compute the entrance of the state according to the list_M 
    eigvals = np.linalg.eigvals(rho_middle)
    # I think I just have to take the density matrix of the subsystem of half of the sites. 
    return eigvals

# Function to plot the eigenvalue density and spacing distribution
def plot_histograms_eigenvalues_density_spacing_distribution(n, L, d):
    eigenvalue_list = []
    spacing_list = []

    for i in range(n):
        eigenvalues = np.log(random_singular_value(d**int(L/2)))
        #vector_state_from_mps(L, d=2)
        eigenvalue_list.extend(eigenvalues)
        
        # Calculate spacing between consecutive eigenvalues
        spacings = np.diff(np.sort(eigenvalues))
        spacing_list.extend(spacings)

    # Plot histogram of eigenvalue density
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(eigenvalue_list, bins=30, density=True, alpha=0.75)
    plt.title("Eigenvalue Density")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    
    # Plot histogram of spacing distribution
    plt.subplot(1, 2, 2)
    plt.hist(spacing_list, bins=30, density=True, alpha=0.75)
    plt.title("Spacing Distribution")
    plt.xlabel("Spacing")
    plt.ylabel("Density")
    
    plt.tight_layout()
    plt.show()
# %%
