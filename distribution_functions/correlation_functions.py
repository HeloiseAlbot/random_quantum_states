import numpy as np
from random_mps import (produce_a_random_mps)
import os 

"""
Code based on the theory from "The density-matrix renormalization group in the age of 
the matrix product states" by Ulrich Schollwöck §4.2.2.
"""

# Pauli-Z matrix
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

def build_transfer_matrix(A, Op):
    """
    Builds the transfer matrix for a single site.
    A: tensor of shape (d, D_left, D_right)
    O: operator of shape (d, d)
    """
    d, Dl, Dr = A.shape
    T = np.zeros((Dl * Dl, Dr * Dr), dtype=np.complex128)
    for s in range(d):
        A_s = A[s]             # shape (Dl, Dr)
        for sp in range(d):
            coeff = Op[sp, s]
            if coeff != 0:
                A_sp_conj = np.conj(A[sp])  # shape (Dl, Dr) 
                kron = np.kron(A_s, A_sp_conj)  # Kronecker product of two arrays. shape (Dl*Dl, Dr*Dr)
                T += coeff * kron
    return T  # shape (Dl^2, Dr^2)

def compute_C_zz(list_A, i, j):
    """
    Compute C^{zz}(i,j) = <psi|Z_i Z_j|psi>
    list_A: list of tensors (d, D_left, D_right)
    """
    L = len(list_A)
    assert 0 <= i < j < L

    env = np.array([[1.0]], dtype=np.complex128)  # scalar identity

    for n in range(0, i):
        T = build_transfer_matrix(list_A[n], I)
        env = env @ T

    T = build_transfer_matrix(list_A[i], Z)
    env = env @ T

    for n in range(i + 1, j):
        T = build_transfer_matrix(list_A[n], I)
        env = env @ T

    T = build_transfer_matrix(list_A[j], Z)
    env = env @ T

    for n in range(j + 1, L):
        T = build_transfer_matrix(list_A[n], I)
        env = env @ T

    return np.trace(env)

def generate_random_MPS(L, d=2):
    """
    Generate a random MPS with physical dimension 2, L sites, and bond dimension D.
    Returns a list of tensors with shape (2, D_left, D_right)
    """
    list_M, _, _, _ = produce_a_random_mps(L, d)
    return list_M

def average_correlation_matrix(L, num_samples):
    """
    Compute averaged C^{zz}(i,j) matrix for all pairs (i, j)
    """
    corr_matrix = np.zeros((L, L))
    for _ in range(num_samples):
        list_M = generate_random_MPS(L)
        for i in range(L):
            for j in range(i + 1, L):
                c = compute_C_zz(list_M, i, j).real
                corr_matrix[i, j] += c
                corr_matrix[j, i] += c  # symmetric
    corr_matrix /= num_samples
    return corr_matrix


#  Example usage:
N = 15
n_samples = 1

all_corr_matrices = np.zeros((n_samples, N, N))

for i in range(n_samples):
    corr_mat = average_correlation_matrix(N, 1)
    all_corr_matrices[i] = corr_mat


# Convert to a 3D NumPy array: shape (n_samples, N, N)
all_corr_matrices = np.array(all_corr_matrices)

# Save to file
np.save(f"corr_matrices_N{N}_n_samples{n_samples}.npy", all_corr_matrices)