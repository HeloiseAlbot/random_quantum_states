import numpy as np
import math

from scipy.optimize import root
from scipy.optimize import minimize
from scipy.linalg import block_diag, eigh, svd
from scipy.stats import unitary_group, norm
from itertools import product
from enum import Enum

class VLevel(Enum):
    QUIET = 0
    INFO = 1
    DEBUG = 2

def vprint(_msg: str, _verbosity: VLevel = VLevel.QUIET, _min_verbosity: VLevel = VLevel.QUIET):
    if (_verbosity.value > VLevel.QUIET.value) and (_verbosity.value >= _min_verbosity.value):
        print(_msg, flush = True)

# Generate random state defined by the level of entanglement between the lattice sites
def random_unitary_matrix(n):
    """
    Generate a random complex matrix of size n * n.
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
 
def compute_sqr_inv(_S):

    if len(_S.shape) == 1:
        return np.array([1.0 / np.sqrt(s) if s > 0.0 else 0.0 for s in _S])
    elif len(_S.shape) == 2:
        return np.diag(np.array([1.0 / np.sqrt(s) if s > 0.0 else 0.0 for s in np.diag(_S)]))
    else:
        raise ValueError("Invalid shape of input matrix: {}".format(_S.shape))

def pseudo_inverse_diag(_S, threshold):
    _S_inv = np.array([1/s if s > threshold else 0.0 for s in _S])
    return _S_inv

def produce_random_mps_v2(L: int, chi_max: int, d: int = 2, _verbosity: int = VLevel.INFO):
    """
    Input:
    L (int): the number of sites in the chain.
    chi_max (int): the bond dimension of the MPS.
    d (int): the dimension of the local Hilbert space.

    Return:
    """

    def get_max_bond_dim(l: int):
        '''
            l counts the bonds starting from the left
        '''
        return min(chi_max, np.power(d, l) if l <= L/2 else np.power(d, L-l))

    def truncate(_S, _threshold):
        """
        Return:
        S: a copy of the matrix _S where the entries which are smaller than _threshold are set to 0. 
        The "normalization of S" is correct (equal to d) for _S diagonal (S is then also diagonal). 
        (Why this choice of normalization ? It fixes the sum of the square of the diag entries). 
        """
        S = np.copy(_S)
        Snorm = np.sum(S**2)
        S[S < _threshold] = 0.0

        return S / np.sum(S**2)

    # def truncated_cholesky(C, chol_threshold):
    #     n = C.shape[0]
    #     # SVD
    #     U, s, Vt = np.linalg.svd(C)
    #     # Keep only singular values above threshold
    #     mask = s > chol_threshold
    #     s_truncated = s[mask]
    #     U_truncated = U[:, mask]
    #     Vt_truncated = Vt[mask, :]
    #     # Reconstruct truncated matrix
    #     C_truncated = U_truncated @ np.diag(s_truncated) @ Vt_truncated
    #     # Cholesky on truncated matrix
    #     L_small = np.linalg.cholesky(C_truncated)
    #     # Now embed L_small into an n×n padded zero matrix
    #     L_padded = np.zeros((n, n))
    #     k = L_small.shape[0]
    #     L_padded[:k, :k] = L_small
    #     return L_padded


    def lowrank_L_padded_with_zeros(C, m, threshold=1e-12):
        n = C.shape[0]
        assert m >= n or m >= 0  # m can be > n; no strict requirement

        # eigendecomposition (symmetric/Hermitian)
        eigvals, eigvecs = np.linalg.eigh(C)
        # keep only positive eigenvalues above threshold
        mask = eigvals > threshold
        eigvals_trunc = eigvals[mask]
        eigvecs_trunc = eigvecs[:, mask]

        # small factor L0 (n x r)
        L0 = eigvecs_trunc @ np.diag(np.sqrt(eigvals_trunc))  # shape (n, r)
        r = L0.shape[1]

        if m < r:
            raise ValueError("m is smaller than the number of retained eigenvalues (r).")
        # pad with zero columns to get shape (n, m)
        L = np.zeros((n, m), dtype=C.dtype)
        L[:, :r] = L0
        return L

    threshold = 1e-8
    list_B = list()
    list_rho = list()

    # Initialization: I need to do it random. Here, it is to start. 
    B_L = np.array([[[1],[0]],[[0],[1]]])
    list_B.append(B_L)

    B_l = B_L
    # Check the end and the start. 
    for l in range(L-1, 0, -1):
        

        left_chi_max = get_max_bond_dim(l) # m_{l-1} in my notations
        right_chi_max = get_max_bond_dim(l+1) # m_{l} in my notations
        print(f"l={l}, m_l={right_chi_max} and m_l-1={left_chi_max}")
        # vprint("Determining Vidal-representation of MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)
        
        # Get next Schmidt-value distribution. The following is the diagonal of eigenvalues.
        S_l_minus_1_just_the_diag = truncate(np.abs(random_eigenvalues_from_n_sphere(left_chi_max)), threshold)
        rho_l_minus_1 = S_l_minus_1_just_the_diag**2 
        assert np.isclose(np.sum(rho_l_minus_1), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))

        list_rho.insert(0, rho_l_minus_1)

        S_l_minus_1 = np.diag(S_l_minus_1_just_the_diag)

        S_l_minus_1_inv = pseudo_inverse_diag(S_l_minus_1_just_the_diag, 1e-4)

        X = random_unitary_matrix(left_chi_max)
        # Check that the columns of X are orthonormal.
        assert np.isclose(np.linalg.norm(np.eye(X.shape[1]) - np.conj(X.T)@X), 0.0)

        # Find a more efficient way than the next line.
        C = (X.conj().T) @ (S_l_minus_1**2) @ X
        U, s, Vt = np.linalg.svd(C)
        if left_chi_max >= right_chi_max: 
            L_lower_tri = U @ np.diag(np.sqrt(s))
        else:
            L_lower_tri = lowrank_L_padded_with_zeros(C, m= d*right_chi_max, threshold=1e-12)

        # L_lower_tri = np.linalg.cholesky(C)
        # L_lower_tri =  truncated_cholesky(C, 1e-8)
        B_l = np.diag(S_l_minus_1_inv) @ X @ L_lower_tri
        
        # now decompose B_l_minus_1 to get actual site tensors
        B_sigma_l = np.zeros([d, left_chi_max, right_chi_max], dtype = complex)
        for idx in range(d):
            B_sigma_l[idx, :, :] = B_l[:,idx*right_chi_max:(idx+1)*right_chi_max] 
            vprint("B_{}@site {}\n{}".format(idx, l, B_sigma_l[idx, :, :]), _verbosity, VLevel.DEBUG)


        list_B.insert(0, np.copy(B_sigma_l))
    return list_B, list_rho


def random_point_on_the_n_sphere(n, r=1):
    # Step 1: Generate the random angles
    angles = np.zeros(n-1)
    angles[:-1] = np.random.uniform(0, np.pi, n - 2)   # φ1, φ2, ..., φ_{n-2} in [0, π]
    angles[-1] = np.random.uniform(0, 2 * np.pi)       # φ_{n-1} in [0, 2π)
    # Step 2: Compute x1, x2, ..., xn
    coordinates = np.zeros(n)
    product_sin = 1  # Initialize the cumulative product of sin terms
    for i in range(n - 1):
        if i > 0:
            product_sin *= np.sin(angles[i - 1])
        coordinates[i] = r * product_sin * np.cos(angles[i])
    # Last coordinate xn with full product of sines
    coordinates[-1] = r * product_sin * np.sin(angles[-1])
    return coordinates
    

def Dirichlet_distribution(n, alpha=1):
    # Step 1: Generate n random numbers from an exponential distribution
    points = np.random.dirichlet(alpha=[alpha] * n)
    return np.sort(points)[::-1]

def squared_moduli_column_Haar_random_unitary(n: int)-> np.array:
    """
    Return the squared modilu of complex elements os a column (the first column) of an
    auxiliary ranodm unitary matrix drawn with respect to the Haar measure on U(n).
    
    Input:
    n (int): the dimension of the vector
    Output:
    psi (np.array): vector of dimension n.
    """
    if n==1:
        return np.array([1])
    else:
        U = unitary_group.rvs(n)
        # Select the first column of the unitary matrix
        psi = U[:, 0]
        # Compute the squared moduli of psi
        squared_moduli = abs(psi)**2
        return squared_moduli

def random_eigenvalues_from_n_sphere(n: int, r=1)-> np.array:
    if n ==1:
        return np.array([[1]])
    else:
        coordinates = random_point_on_the_n_sphere(n, r)
        return coordinates
        # return np.sort(np.square(coordinates))[::-1]