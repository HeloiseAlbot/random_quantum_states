import numpy as np
import math

# from scipy.optimize import root
# from scipy.optimize import minimize
from scipy.linalg import eigh, svd, block_diag, cholesky, solve_triangular
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

def draw_random_mps(length: int, chi_max: int, d: int = 2, standard_deviation: float = None, _verbosity: int = VLevel.INFO):
    """
    Input:
    
    length (int): the number of sites in the chain.
    chi_max (int): the bond dimension of the MPS.
    d (int): the dimension of the local Hilbert space.
    standard_deviation (float): if it is None, the eigenvalues are uniformly distributed on the unit n-sphere.
    Otherwise, the eigenvalues are distributed around the maximally entangled point following a Gaussian distribution of stardard deviation = standard_deviation. 

    Return:
    list_M (list[np.array]): list of the tensor of the MPS.
    list_rho (list[np.array]): list of the array of the eigenvalues between the different bipartition obtained with one cut on the lattice chain.
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
            # s_errors.append(np.linalg.norm(np.sort(s)[::-1] - list_S[bond_idx]))
            s_errors.append(np.linalg.norm(np.square(np.sort(s)[::-1]) - np.square(list_S[bond_idx])))

        # list_errors.append(np.linalg.norm(s_errors)/len(s_errors))
        list_errors.append(np.sum(s_errors)/len(s_errors))
        vprint("it = {}, ||δS|| = {}".format(curr_it, list_errors[-1]), _verbosity, VLevel.DEBUG)

        # increase counter
        curr_it += 1

    return list_M, list_errors

def random_point_on_the_n_sphere(n, r=1): 
    """
    Return the coordinates of random points uniformly distributed on the n-sphere.
    By default, r=1 so it is the unit n-sphere.
    """
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
    Return the squared modilu of complex elements of a column (the first column) of an
    auxiliary random unitary matrix drawn with respect to the Haar measure on U(n).
    
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

def vector_state_from_mps_v2(list_M: list, L: int, d: int = 2) -> np.array:
    """
    Input:
    list_M (list): List of L rank-3 tensors with shape (d, chi_left, chi_right)
    L (int): The number of lattice sites. 
    d (int): The local physical dimension of each site.
    
    Output:
    state_flat (np.array): A complex 1D array of size d**L representing the full state vector.
    sigma_tuples (list): Combinations of all physical index states.
    """
    # Create the target state vector (flat)
    num_configurations = d ** L
    state_flat = np.zeros(num_configurations, dtype=complex)
    
    # Generate all possible configurations (e.g., (0,0,0), (0,0,1)...)
    sigma_tuples = list(product(range(d), repeat=L))
    
    # Iterate over every configuration in the Hilbert space
    for sidx, sigma_tuple in enumerate(sigma_tuples):
        
        # Start contraction with the physical index of the first site
        # This yields a matrix of shape (1, chi_1)
        psi_config = list_M[0][sigma_tuple[0], :, :]
        
        # Sequentially multiply the matrices for the remaining sites
        for l in range(1, L):
            psi_config = psi_config @ list_M[l][sigma_tuple[l], :, :]
        
        # Since the boundaries are open (chi_left[0] = 1 and chi_right[-1] = 1),
        # psi_config is now a 1x1 matrix. Extract the scalar amplitude.
        state_flat[sidx] = psi_config[0, 0]

    return state_flat, sigma_tuples

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



