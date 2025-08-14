import numpy as np
from random_mps import (produce_random_mps, produce_random_mps_eigenvalues_in_a_disk)
import os 

"""
Code based on the theory from "The density-matrix renormalization group in the age of 
the matrix product states" by Ulrich Schollwöck §4.2.1.
"""

# Pauli-Z matrix
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

def contract1(A, C, Op=None):
    """
    WRITE WHAT IT DOES.
    A: tensor of shape (d, D_left, D_right)
    C: matrix of dimension (D_right, D_right)
    O: operator of shape (d, d)
    """
    d, Dl, Dr = A.shape
    new_C = np.zeros((Dr, Dr),  dtype=np.complex128)
    if Op is None:
        for s in range(d):
            A_s = A[s]
            A_s_dagger = A_s.conj().T  # shape (Dr, Dl)
            new_C += A_s_dagger @ C @ A[s] 
        return new_C
    else: # Apply the operator
        for s in range(d):
            A_s = A[s]
            A_s_dagger = A_s.conj().T  # shape (Dr, Dl)
            for sp in range(d):
                new_C +=  Op[s,sp] * A_s_dagger @ C @ A[sp]
        return new_C # shape (Dl, Dl)

def compute_C_zz2(list_A, i, j):
    """
    Compute C^{zz}(i,j) = <psi|Z_i Z_j|psi>
    list_A: list of tensors (d, D_left, D_right)
    """
    L = len(list_A)
    assert 0 <= i < j < L

    C = np.array([[1.0]], dtype=np.complex128)  # scalar identity

    for n in range(0, i):
        C = contract1(list_A[n], C, None)

    C = contract1(list_A[i], C, Z)

    for n in range(i + 1, j):
        C = contract1(list_A[n], C, None)

    C = contract1(list_A[j], C, Z)

    for n in range(j + 1, L):
        C = contract1(list_A[n], C, None)

    return C


def average_correlation_matrix2(L: int, chi_max:int, d: int=2, num_samples: int=1, diameter_disk=None):
    """
    Compute averaged C^{zz}(i,j) matrix for all pairs (i, j)
    """
    corr_matrix = np.zeros((L, L))
    if diameter_disk is None:
        for _sample in range(num_samples):
            list_M, _ = produce_random_mps(L, chi_max, d)
            for i in range(L):
                for j in range(i + 1, L):
                    c = compute_C_zz2(list_M, i, j).real
                    corr_matrix[i, j] += c
                    corr_matrix[j, i] += c  # symmetric
        corr_matrix /= num_samples
        return corr_matrix
    else:
        for _sample in range(num_samples):
            list_M, _ = produce_random_mps_eigenvalues_in_a_disk(L, chi_max, diameter_disk, d)
            for i in range(L):
                for j in range(i + 1, L):
                    c = compute_C_zz2(list_M, i, j).real
                    corr_matrix[i, j] += c
                    corr_matrix[j, i] += c  # symmetric
        corr_matrix /= num_samples
        return corr_matrix




#  Example usage:
N = 10
n_samples = 100
d = 2 # Dimension of the local Hilbert space. 
standard_derivation = 0.25
all_corr_matrices = np.zeros((n_samples, N, N))

for i in range(n_samples):
    corr_mat = average_correlation_matrix2(N, d, 1, standard_derivation)
    all_corr_matrices[i] = corr_mat


# Convert to a 3D NumPy array: shape (n_samples, N, N)
all_corr_matrices = np.array(all_corr_matrices)

# Save to file
np.save(f"corr_matrices_N{N}_n_samples{n_samples}_standard_deviation{standard_derivation}.npy", all_corr_matrices)
# np.save(f"corr_matrices_N{N}_n_samples{n_samples}.npy", all_corr_matrices)