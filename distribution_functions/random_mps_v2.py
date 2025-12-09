import numpy as np
import math

from scipy.optimize import root
from scipy.optimize import minimize
from scipy.linalg import block_diag, eigh, svd, cholesky, solve_triangular
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
        count_below = np.sum(S < _threshold)
        print("Number of entries below threshold:", count_below)
        S[S < _threshold] = 0.0 

        return S / np.sum(S**2), count_below

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


    # def lowrank_L_padded_with_zeros(C, m, threshold=1e-12):
    #     n = C.shape[0]
    #     assert m >= n or m >= 0  # m can be > n; no strict requirement

    #     # eigendecomposition (symmetric/Hermitian)
    #     eigvals, eigvecs = np.linalg.eigh(C)
    #     # keep only positive eigenvalues above threshold
    #     mask = eigvals > threshold
    #     eigvals_trunc = eigvals[mask]
    #     eigvecs_trunc = eigvecs[:, mask]

    #     # small factor L0 (n x r)
    #     L0 = eigvecs_trunc @ np.diag(np.sqrt(eigvals_trunc))  # shape (n, r)]
    #     L0_lower_tri, Q_L0 = np.linalg.cholesky(L0)
    #     print(f"L0:{L0}")
    #     r = L0.shape[1]

    #     if m < r:
    #         raise ValueError("m is smaller than the number of retained eigenvalues (r).")
    #     # pad with zero columns to get shape (n, m)
    #     L = np.zeros((n, m), dtype=C.dtype)
    #     L[:, :r] = L0
    #     print(f"L:{L}")
    #     return L

    threshold = 1e-8
    list_M = list()
    list_rho = list()
    
    list_rho.append(np.array([[1.]]))

    # Check the end and the start. 
    for l in range(L-1, int(L/2-0.5), -1):
        left_chi_max = get_max_bond_dim(l) # m_{l-1} in my notations
        right_chi_max = get_max_bond_dim(l+1) # m_{l} in my notations
        # print(f"l={l}, m_l={right_chi_max} and m_l-1={left_chi_max}")
        vprint("Determining Vidal-representation of MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)
        
        # Get next Schmidt-value distribution. The following is the diagonal of eigenvalues.
        S_l_minus_1_just_the_diag, count_below = truncate(np.abs(random_eigenvalues_from_n_sphere(left_chi_max)), threshold)

        rho_l_minus_1 = S_l_minus_1_just_the_diag**2 
        assert np.isclose(np.sum(rho_l_minus_1), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))

        list_rho.insert(0, rho_l_minus_1)

        S_l_minus_1 = np.diag(S_l_minus_1_just_the_diag)
        S_l_minus_1_inv_just_the_diag = pseudo_inverse_diag(S_l_minus_1_just_the_diag, 1e-4)

        X = random_unitary_matrix(left_chi_max)
        # Check that the columns of X are orthonormal.
        assert np.isclose(np.linalg.norm(np.eye(X.shape[1]) - np.conj(X.T)@X), 0.0)

        # See my notes to see why the coefficient (d**2) appears.
        C = (d**2)*(X.conj().T) @ (S_l_minus_1**2) @ X
        L_lower_tri = np.linalg.cholesky(C)
        # if count_below == 0: # C is positive definite.
        #     L_lower_tri = np.linalg.cholesky(C)
        # else: # C is not positive definite.
        #     L0_lower_tri = np.linalg.cholesky(C)
        #     L_lower_tri = np.zeros([left_chi_max, d * right_chi_max])
        #     _, r = L0_lower_tri.shape
        #     L_lower_tri[:,:r] = L0_lower_tri
        # try:
            # L = np.linalg.cholesky(C)   # L is lower-triangular, C = L @ L.conj().T
        # except np.linalg.LinAlgError:
            # Do something



        # print(f"L_lower_tri: {L_lower_tri}")

        # L_lower_tri =  truncated_cholesky(C, 1e-8)
        # if l==0:
        #     diag_s = np.diag(S_l_minus_1_inv_just_the_diag)
        #     diag_s = diag_s.reshape(1,1)
        #     B_l = diag_s @ X @ L_lower_tri
        # else:
        # See my notes to see why the coefficient (1/d) appears.
        B_l = (1/d)*np.diag(S_l_minus_1_inv_just_the_diag) @ X @ L_lower_tri
        assert is_unitary(B_l), "Matrix is not unitary: A†A deviates from identity." 

        # now decompose B_l_minus_1 to get actual site tensors
        B_sigma_l = np.zeros([d, left_chi_max, right_chi_max], dtype = complex)
        for idx in range(d):
            B_sigma_l[idx, :, :] = B_l[:,idx*right_chi_max:(idx+1)*right_chi_max] 
            vprint("B_{}@site {}\n{}".format(idx, l, B_sigma_l[idx, :, :]), _verbosity, VLevel.DEBUG)
        list_M.insert(0, np.copy(B_sigma_l))
    for l in range(0, int(L/2), 1):
        left_chi_max = get_max_bond_dim(l) # m_{l-1} in my notations
        right_chi_max = get_max_bond_dim(l+1) # m_{l} in my notations
        # print(f"l={l}, m_l={right_chi_max} and m_l-1={left_chi_max}")
        vprint("Determining Vidal-representation of MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)
        
        # Get next Schmidt-value distribution. The following is the diagonal of eigenvalues.
        S_l_plus_1_just_the_diag, count_below = truncate(np.abs(random_eigenvalues_from_n_sphere(right_chi_max)), threshold)

        rho_l_plus_1 = S_l_plus_1_just_the_diag**2 
        assert np.isclose(np.sum(rho_l_plus_1), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))

        list_rho.insert(l, rho_l_plus_1)

        S_l_plus_1 = np.diag(S_l_plus_1_just_the_diag)
        S_l_plus_1_inv_just_the_diag = pseudo_inverse_diag(S_l_plus_1_just_the_diag, 1e-4)

        X = random_unitary_matrix(right_chi_max)
        # Check that the columns of X are orthonormal.
        assert np.isclose(np.linalg.norm(np.eye(X.shape[1]) - np.conj(X.T)@X), 0.0)

        # See my notes to see why the coefficient (d**2) appears.
        C = (d**2)*X @ (S_l_plus_1**2) @ (X.conj().T)

        L_lower_tri = np.linalg.cholesky(C)
        # if count_below == 0: # C is positive definite.
        #     L_lower_tri = np.linalg.cholesky(C)
        # else: # C is not positive definite.
        #     L0_lower_tri = np.linalg.cholesky(C)
        #     L_lower_tri = np.zeros([left_chi_max, d * right_chi_max])
        #     _, r = L0_lower_tri.shape
        #     L_lower_tri[:,:r] = L0_lower_tri
        # try:
            # L = np.linalg.cholesky(C)   # L is lower-triangular, C = L @ L.conj().T
        # except np.linalg.LinAlgError:
            # Do something
            # L_lower_tri = lowrank_L_padded_with_zeros(C, m= d*right_chi_max, threshold=1e-8)


        # print(f"L_lower_tri: {L_lower_tri}")

        # L_lower_tri =  truncated_cholesky(C, 1e-8)
        # if l==0:
        #     diag_s = np.diag(S_l_minus_1_inv_just_the_diag)
        #     diag_s = diag_s.reshape(1,1)
        #     B_l = diag_s @ X @ L_lower_tri
        # else:
        # See my notes to see why the coefficient (1/d) appears.
        A_l_plus_1 = (1/d)*((L_lower_tri.conj().T)) @ X @ np.diag(S_l_plus_1_inv_just_the_diag) 
        assert is_unitary(A_l_plus_1), "Matrix is not unitary: A†A deviates from identity."

 

        # now decompose A_l_plus_1 to get actual site tensors
        A_sigma_l_plus_1 = np.zeros([d, left_chi_max, right_chi_max], dtype = complex)
        for idx in range(d):
            A_sigma_l_plus_1[idx, :, :] = A_l_plus_1[idx*left_chi_max:(idx+1)*left_chi_max, :] 
            vprint("B_{}@site {}\n{}".format(idx, l, A_sigma_l_plus_1[idx, :, :]), _verbosity, VLevel.DEBUG)
        list_M.insert(l, np.copy(A_sigma_l_plus_1))


    return list_M, list_rho

def draw_random_mps(length: int, chi_max: int, d: int = 2, standard_deviation: int = None, _verbosity: int = VLevel.INFO):
    """
    Input:
    length (int): the number of sites in the chain.
    chi_max (int): the bond dimension of the MPS.
    d (int): the dimension of the local Hilbert space.

    Return:
    """

    def get_max_bond_dim(l: int):
        """
            l counts the bonds starting from the left
        """
        return min(chi_max, np.power(d, l) if l < length/2 else np.power(d, length-l))

    def truncate(_S, _threshold):
        """
        Return: a copy of the singular value matrix matrix _S where the entries which are smaller than _threshold are set to 0. 
        """
        S = np.copy(_S)
        S[S < _threshold] = 0.0
        
        return S / np.sqrt(np.sum(S**2))

    def pseudo_inverse_diag(_S, _threshold):
        return np.array([1.0 / s if s > _threshold else 0.0 for s in _S])

    def check_majorization(_S1, _S2, l):
        """
        If l <= length/2:
            Returns true if _S2 weakly majorizes _S1
        else: (l > length/2)
            Returns true if _S1 weakly majorizes _S2
        """
        if isinstance(_S1, np.ndarray):
            assert len(_S1.shape) == 1
        if isinstance(_S2, np.ndarray):
            assert len(_S2.shape) == 1

        max_len = len(_S1)

        if (l > length/2):
            _S1, _S2 = _S2, _S1

        summed_S1 = np.cumsum(np.sort(_S1)[::-1][:max_len]**2)
        summed_S2 = np.cumsum(np.sort(_S2)[::-1][:max_len]**2)

        return np.all(summed_S1 <= summed_S2)

    threshold = 1e-08
    prev_S = np.array([1.0])
    prev_V = np.array([[1.0]])
    list_M = list()
    list_rho = list()

    if standard_deviation is None:
        function_eigenvalues = random_eigenvalues_from_n_sphere
    else:
        def function_eigenvalues(_right_chi_max):
            angles_center_l = generate_angle_maximally_entangled_eigenvalues(_right_chi_max)
            _eigenvalues = gaussian_point_on_the_n_sphere_in_a_disk(_right_chi_max, standard_deviation, angles_center_l, r=1)
            return _eigenvalues

    for l in range(length):

        left_chi_max = get_max_bond_dim(l)
        right_chi_max = get_max_bond_dim(l+1)

        vprint("Sampling MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)

        if l < length-1:

            '''
            We sample Q with dim (d*left_chi_max, d*left_chi_max)
            '''
            Q = random_unitary_matrix(d*left_chi_max)[:,:right_chi_max]
            assert np.isclose(np.linalg.norm(np.eye(Q.shape[1]) - np.conj(Q.T)@Q), 0.0)

            # Get next Schmidt-value distribution ensuring majorization constraint
            # success = False
            # max_tries = 2048
            # max_spectrum = np.kron(prev_S, np.ones(d))
            # for idx in range(max_tries):
            # for idx in range(1):
            next_S = truncate(np.sort(np.abs(function_eigenvalues(right_chi_max)))[::-1], threshold)
            # if check_majorization(next_S, max_spectrum, l):
            #     success = True
            #     break

            # if not(success):
            #     print("criteria majorization not okay")

            #     # raise ValueError("Failure satisfying majorization criterion for sampled Schmidt values after {} attempts".format(max_tries))
            # else:
            #     vprint("After {} iterations found next set of singular values satisfying majorization constraint".format(idx), _verbosity, VLevel.INFO)

            rho = next_S**2
            assert np.isclose(np.sum(rho), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))
            list_rho.append(rho)

            # construct X = ΣWQ and compute SVD
            SigmaW = np.kron(np.eye(d), np.diag(prev_S) @ prev_V)
            w,s,zh = svd(SigmaW @ Q)

            # Now evaluate eigenvalue decomposition of z^+ s^{-1} next_S^2 s^{-1} z 
            S = np.conj(zh).T @ np.diag((next_S * pseudo_inverse_diag(s,1e-08))**2) @ zh
            E, V = eigh((S + np.conj(S).T)/2.0)
            E[E<0] = 0.0
            R = V @ np.diag(np.sqrt(E)) @ np.conj(V.T)

            # and construct M
            M = Q @ R

            # compute right basis trafo to be passed to next site and check singular values
            u,s,next_V = svd(SigmaW @ M)
            assert np.isclose(np.linalg.norm(np.sort(s) - np.sort(next_S)), 0.0, rtol = 1e-08)

            # construct and decompose site tensor
            M_l = np.zeros([d, left_chi_max, right_chi_max], dtype = complex)
            for idx in range(d):
                M_l[idx, :, :] = M[idx*left_chi_max:(idx+1)*left_chi_max,:]
                vprint("M_{}@site {}\n{}".format(idx, l, M_l[idx, :, :]), _verbosity, VLevel.DEBUG)

            prev_S = next_S
            prev_V = next_V

        else:

            V = random_unitary_matrix(d*right_chi_max)[:left_chi_max,:]
            assert np.isclose(np.linalg.norm(np.eye(V.shape[0]) - V@np.conj(V.T)), 0.0)

            M_l = np.zeros([d, left_chi_max, right_chi_max], dtype = complex)
            for idx in range(d):
                M_l[idx, :, :] = V[:,idx*right_chi_max:(idx+1)*right_chi_max]
                vprint("M_{}@site {}\n{}".format(idx, l, M_l[idx, :, :]), _verbosity, VLevel.DEBUG)

        list_M.append(np.copy(M_l))

    return list_M, list_rho

def iterative_solve_mps(list_M: list, list_rho: list, maxit: int = 100, eps: float = 1e-06, _verbosity: int = VLevel.INFO):

    def fuse(M_l: np.ndarray, right_sweep: bool):

        d = M_l.shape[0]
        left_bdim = M_l.shape[1]
        right_bdim = M_l.shape[2]

        if right_sweep:
            result = np.zeros([d * left_bdim, right_bdim], dtype = complex)
            for s in range(d):
                result[s*left_bdim:(s+1)*left_bdim,:] = np.copy(M_l[s,:,:])
        else:
            result = np.zeros([left_bdim, d*right_bdim], dtype = complex)
            for s in range(d):
                result[:,s*right_bdim:(s+1)*right_bdim] = np.copy(M_l[s,:,:])

        return result

    def split(M: np.ndarray, d: int, left_bdim: int, right_bdim: int, right_sweep: bool):

        result = np.zeros([d, left_bdim, right_bdim], dtype = complex)
        for s in range(d):
            if right_sweep:
                result[s,:,:] = np.copy(M[s*left_bdim:(s+1)*left_bdim,:])
            else:
                result[s,:,:] = np.copy(M[:,s*right_bdim:(s+1)*right_bdim])

        return result

    def absorb(M_l: np.ndarray, t: np.ndarray, right_sweep: bool):

        d = M_l.shape[0]

        result = np.zeros(M_l.shape, dtype = complex)
        for s in range(d):
            if right_sweep:
                result[s,:,:] = t @ M_l[s,:,:]
            else:
                result[s,:,:] = M_l[s,:,:] @ t

        return result

    n_sites = len(list_M)
    curr_it = 0
    list_S = [np.sort(np.sqrt(evs))[::-1] for evs in list_rho]
    list_errors = []

    # do a warmup sweep to bring mps into mixed-canonical form with orthogonality center at last site
    right_sweep = True
    for i in range(n_sites-1):

        u,s,v = svd(fuse(list_M[i], right_sweep), full_matrices = False)
        list_M[i] = split(u, *list_M[i].shape, right_sweep)
        list_M[i+1] = absorb(list_M[i+1], np.diag(s)@v, right_sweep)

    while (curr_it < maxit) and ((len(list_errors) == 0) or (list_errors[-1] > eps)):

        s_errors = []

        # flip sweep direction
        right_sweep = not(right_sweep)

        for i in range(n_sites-1):

            site_idx = i if right_sweep else n_sites - i - 1
            bond_idx = site_idx if right_sweep else site_idx - 1

            u,s,v = svd(fuse(list_M[site_idx], right_sweep), full_matrices = False)

            if right_sweep:
                list_M[site_idx] = split(u, *list_M[site_idx].shape, right_sweep)
                list_M[site_idx+1] = absorb(list_M[site_idx+1], np.diag(list_S[bond_idx])@v, right_sweep)
            else:
                list_M[site_idx] = split(v, *list_M[site_idx].shape, right_sweep)
                list_M[site_idx-1] = absorb(list_M[site_idx-1], u @ np.diag(list_S[bond_idx]), right_sweep)
            # v = np.linalg.norm(np.sort(s)[::-1] - list_S[bond_idx]) 
            # result = np.linalg.norm(v) / v.size    
            s_errors.append(np.linalg.norm(np.sort(s)[::-1] - list_S[bond_idx]))

        list_errors.append(np.linalg.norm(s_errors)/len(s_errors))
        vprint("it = {}, ||δS|| = {}".format(curr_it, list_errors[-1]), _verbosity, VLevel.DEBUG)

        # increase counter
        curr_it += 1

    return list_M, list_errors

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


def vector_state_from_mps_v2(list_M: list, list_rho: list, L:int, d=2)->np.array:
    """
    Input:
    list_M (list): list of the matrices M_sigma_l ordered increasingly. The matrices M_sigma_l[l] for l
    in [1, L/2-1] contained the matrices A_l^{\sigma_l} are stored. The matrices M_sigma_l[l] for l
    in [L/2, L] contained the matrices B_l^{\sigma_l} are stored.
    list_rho (list): list of the eigenvalues ordered increasingly.
    L (int): the number of lattice sites. 
    d (int): The size of the sigma_l index, the dimension of the qudit.
    
    Output:
    vector_state (np.array): vector_state[sigma_1, sigma_2, ..., sigma_L] holds the desired matrix product
    """
    # Initialize vector_state with shape (d, d, ..., d) repeated L times
    vector_state_shape = (d,) * L
    vector_state = np.zeros(vector_state_shape, dtype=object)
    vector_labels = np.zeros(vector_state_shape, dtype=object)
    sigma_tuples = list(product(range(d), repeat=L))
    rho_middle = np.diag(list_rho[int(L/2+1)])
    # Iterate over all combinations of sigma_1, sigma_2, ..., sigma_L
    for sidx, sigma_tuple in enumerate(sigma_tuples):
        # Initialize product as identity matrix of size (n x n)
        product_matrix = np.eye(1, dtype="complex")

        # Multiply A_1^{sigma_1} @ A_2^{sigma_2} @ ... @ A_L^{sigma_L}
        for l, sigma_l in enumerate(sigma_tuple):
            A = list_M[l]
            product_matrix = product_matrix @ A[sigma_l, :, :]
            if l==int(L/2):
                # Eq. (53) of The DMRG in the age of MPS by US. 
                product_matrix = product_matrix @ np.sqrt(rho_middle)
        

        # Assign the result to vector_state at the index sigma_1, ..., sigma_L
        vector_state[sigma_tuple] = product_matrix
        vector_labels[sigma_tuple] = sidx

    # Now vector_state[sigma_1, sigma_2, ..., sigma_L] holds the desired matrix product
    vector_state = vector_state.reshape((d**L))
    state_flat = np.array([x[0] for x in vector_state])
    labels_flat = vector_labels.reshape((d**L))
    return state_flat.reshape(d**L), sigma_tuples, labels_flat

def is_unitary(A, tol=1e-10):
    A = np.asarray(A)
    I = np.eye(A.shape[0])
    return np.allclose(A.conj().T @ A, I, atol=tol) and \
           np.allclose(A @ A.conj().T, I, atol=tol)

def generate_angle_maximally_entangled_eigenvalues(n: int)-> np.array:
    """
    Generate the spherical coordinates corresponding to the Cartesian coordinates given by 
    (1/\sqrt{n},...,1/\sqrt{n}).
    
    Return:
    phi_i (np.array of shape (n-1)): array of angles (in radian).
    """
    phi_i = np.arccos([1/math.sqrt(n+1-i) for i in range(1,n)])
    return phi_i


def gaussian_point_on_the_n_sphere_in_a_disk(n: int, standard_deviation: float, angles_center: np.array, r: float=1):
    """
    Input:
    n (int): dimension of the space of the n-sphere. We should have angles_center.size = n-1. 
    standard_deviation (float): the standard deviation of the Gaussian distribution (in radian). 
                    We DO NOT place the simulation in the positive orthant. 
                    Then, in order to keep a uniform distribution on the disk of the n-sphere we should have: 
                    epsilon_over_2 > math.asin(1/math.sqrt(n)), hence the ValueError line. 
    angles_center (np.array): coordinates of the point with is the center of the disk in which are taken 
                    the random point. By default, it will be the point giving the maximally mixed states. 
    r (float): rayon of the n-sphere. By default, 1.

    Output: 
    new_coordinates (np.array): A random point distributed as a Gaussian around the point whose coordinates 
                    are angles_center with standard deviation standard_deviation. The point is given via its 
                    Cartesian coordinates. 
    """
    if angles_center.size != n-1:
        raise ValueError(f"The input array angles_center must be of size n-1.")

    # if rayon_disk > min(math.asin(1/math.sqrt(n)), np.pi/4):
    #     raise ValueError(f"The input value diameter_disk must be smaller than {2*math.asin(1/math.sqrt(n))} and pi/2")
    # Generate the coordinates of the maximal
    if angles_center is None:
        print("Angles center is None")
        angles_center = generate_angle_maximally_entangled_eigenvalues(n)
    # Step 1: Generate the random angles
    delta_angles =  np.random.normal(0, standard_deviation, n-1)    
    # Step 2: Compute x1, x2, ..., xn
    new_angles = angles_center + delta_angles
    new_coordinates = np.zeros(n)
    product_sin = 1  # Initialize the cumulative product of sin terms
    for i in range(n - 1):
        if i > 0:
            product_sin *= np.sin(new_angles[i - 1])
        new_coordinates[i] = r * product_sin * np.cos(new_angles[i])
    # Last coordinate xn with full product of sines
    if n == 1:
        new_coordinates[-1] = r * product_sin
    else:
        new_coordinates[-1] = r * product_sin * np.sin(new_angles[-1])
    return new_coordinates



