import numpy as np
from distribution_functions.random_mps_v2 import produce_random_mps, produce_random_mps_eigenvalues_in_a_disk, produce_random_mps_eigenvalues_gaussian_in_a_disk

# Pauli-Z matrix
Z = np.array([[1, 0], [0, -1]])

def contract1(A, C, Op=None):
    """
    WRITE WHAT IT DOES.
    A: tensor of shape (d, D_left, D_right)
    C: matrix of dimension (D_right, D_right)
    O: operator of shape (d, d)
    """
    d, Dl, Dr = A.shape
    new_C = np.zeros((Dr, Dr), dtype=np.complex128)
    if Op is None:
        for s in range(d):
            A_s = A[s]
            A_s_dagger = A_s.conj().T
            new_C += A_s_dagger @ C @ A[s]
    else:
        for s in range(d):
            A_s = A[s]
            A_s_dagger = A_s.conj().T
            for sp in range(d):
                new_C += Op[s, sp] * A_s_dagger @ C @ A[sp]
    return new_C

def compute_expectation(list_A, i, Op):
    """Compute ⟨ψ| Op_i |ψ⟩"""
    L = len(list_A)
    C = np.array([[1.0]], dtype=np.complex128)
    for n in range(0, i):
        C = contract1(list_A[n], C, None)
    C = contract1(list_A[i], C, Op)
    for n in range(i + 1, L):
        C = contract1(list_A[n], C, None)
    return np.trace(C).real

def compute_C_zz_connected_cached(list_A, i, j, cache):
    """
    Compute connected ⟨σ^z_i σ^z_j⟩ - ⟨σ^z_i⟩⟨σ^z_j⟩ using cache for ⟨σ^z_k⟩
    """
    if i not in cache:
        cache[i] = compute_expectation(list_A, i, Z)
    if j not in cache:
        cache[j] = compute_expectation(list_A, j, Z)

    exp_i = cache[i]
    exp_j = cache[j]

    # Two-point function ⟨σ^z_i σ^z_j⟩
    L = len(list_A)
    C = np.array([[1.0]], dtype=np.complex128)
    for n in range(0, i):
        C = contract1(list_A[n], C, None)
    C = contract1(list_A[i], C, Z)
    for n in range(i + 1, j):
        C = contract1(list_A[n], C, None)
    C = contract1(list_A[j], C, Z)
    for n in range(j + 1, L):
        C = contract1(list_A[n], C, None)

    exp_ij = np.trace(C).real
    return exp_ij - exp_i * exp_j

def average_connected_C_zz_uniform(L: int, chi_max: int, num_samples:int = 1, d:int = 2):
    """
    Compute averaged connected C^{zz}(i,j) matrix for all i < j
    """
    corr_matrix = np.zeros((L, L))
    for _ in range(num_samples):
        list_M, _ =  produce_random_mps(L, chi_max, d)
        cache = {}  # Cache of ⟨σ^z_i⟩
        for i in range(L):
            for j in range(i + 1, L):
                c = compute_C_zz_connected_cached(list_M, i, j, cache)
                corr_matrix[i, j] += c
                corr_matrix[j, i] += c
    corr_matrix /= num_samples
    return corr_matrix

def average_connected_C_zz_eigenvalues_in_a_disk(L: int, chi_max: int, diameter_disk: float, num_samples:int = 1, d:int = 2):
    """
    Compute averaged connected C^{zz}(i,j) matrix for all i < j
    """
    corr_matrix = np.zeros((L, L))
    for _ in range(num_samples):
        list_M, _ =  produce_random_mps_eigenvalues_in_a_disk(L, chi_max, diameter_disk, d)
        cache = {}  # Cache of ⟨σ^z_i⟩
        for i in range(L):
            for j in range(i + 1, L):
                c = compute_C_zz_connected_cached(list_M, i, j, cache)
                corr_matrix[i, j] += c
                corr_matrix[j, i] += c
    corr_matrix /= num_samples
    return corr_matrix

def average_connected_C_zz_eigenvalues_gaussian_in_a_disk(L: int, chi_max: int, standard_deviation: float, num_samples:int = 1, d:int = 2):
    """
    Compute averaged connected C^{zz}(i,j) matrix for all i < j
    """
    corr_matrix = np.zeros((L, L))
    for _ in range(num_samples):
        list_M, _ = produce_random_mps_eigenvalues_gaussian_in_a_disk(L, chi_max, standard_deviation, d)

        cache = {}  # Cache of ⟨σ^z_i⟩
        for i in range(L):
            for j in range(i + 1, L):
                c = compute_C_zz_connected_cached(list_M, i, j, cache)
                corr_matrix[i, j] += c
                corr_matrix[j, i] += c
    corr_matrix /= num_samples
    return corr_matrix

#  Example usage:
L = 20
n_samples = 100
chi_max = 64
d = 2 # Dimension of the local Hilbert space. 
standard_deviation = 0.001 # standard_deviation = 0.321750554396642
all_corr_matrices = np.zeros((n_samples, L, L))

for i in range(n_samples):
    # corr_mat = average_connected_C_zz_uniform(L, chi_max, 20, 2)
    corr_mat = average_connected_C_zz_eigenvalues_gaussian_in_a_disk(L, chi_max, standard_deviation, n_samples, d)

    
    all_corr_matrices[i] = corr_mat


# Convert to a 3D NumPy array: shape (n_samples, N, N)
all_corr_matrices = np.array(all_corr_matrices)

# Save to file
# np.save(f"connected_corr_matrices_L{L}_n_samples{n_samples}_chi_max{chi_max}.npy", all_corr_matrices)
np.save(f"connected_corr_matrices_L{L}_n_samples{n_samples}_chi_max{chi_max}_standard_deviation{standard_deviation}.npy", all_corr_matrices)
