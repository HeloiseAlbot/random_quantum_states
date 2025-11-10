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

def recursive_pseudo_inverse(M, threshold, _verbosity: int = VLevel.INFO, lapack_driver = 'gesvd'):

    U, S, Vh = svd(M, lapack_driver = lapack_driver)

    # First sort singular values to ensure decreasing singular values
    sort_idxs = np.argsort(S)[::-1]
    U = U[:, sort_idxs]
    S = S[sort_idxs]
    Vh = Vh[sort_idxs,:]

    # Now look for threshold singular value
    i_threshold = len(S)
    for idx in range(i_threshold):
        if S[idx]/S[0] < threshold:
            i_threshold = idx
            break

    S_sqr_inv = compute_sqr_inv(S)

    # If there are singular valus below threshold, recursively call svd for this subspace
    if i_threshold < len(S):

        vprint("In recursive svd, entering sublayer at idx {} of {} singular values".format(i_threshold, len(S)), _verbosity, VLevel.DEBUG)

        U_reduced = U[:,i_threshold:]
        Vh_reduced = Vh[i_threshold:,:]
        M_reduced = np.conj(U_reduced).T @ (M @ np.conj(Vh_reduced).T)

        U_prime, S_sqr_inv_prime, Vh_prime = recursive_pseudo_inverse(M_reduced / threshold , threshold)

        U[:,i_threshold:] = U[:,i_threshold:] @ U_prime
        S_sqr_inv[i_threshold:] = S_sqr_inv_prime / np.sqrt(threshold)
        Vh[i_threshold:,:] = Vh_prime @ Vh[i_threshold:,:]

    return U, S_sqr_inv, Vh

def produce_random_mps(L: int, chi_max: int, d: int = 2, _verbosity: int = VLevel.INFO):
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

    def solve_inversion(_D, _Q):
        """
        Return:
        A = Q^+ S^2 Q
        """

        A = np.conj(_Q.T) @ np.kron(np.eye(d), np.diag(_D)) @ _Q

        vprint("Solving inverse square root of A = Q^+ S^2 Q, dim(A) = {} x {}".format(*A.shape), _verbosity, VLevel.DEBUG)
        U, Sigma_sqr_inv, Vh = recursive_pseudo_inverse(A, 1e-4, _verbosity = _verbosity)

        assert np.all(np.isclose(Sigma_sqr_inv.imag, 0.0)), "non-vanishing imaginary part in singular values detected, violated condition is np.isclose(S.imag, 0.0) = {}".format(np.isclose(S.imag, 0.0))

        if np.any(Sigma_sqr_inv > 1e7):
            vprint("Warning, numerical threshold reached during inversion, consider using smaller bond dimensions or narrow eigenvalue distribution", _verbosity, VLevel.INFO)

        return np.conj(Vh.T) @ np.diag(Sigma_sqr_inv) @ np.conj(U.T)


    threshold = 1e-16
    prev_S = np.array([1.0])
    list_M = list()
    list_rho = list()

    for l in range(L):

        left_chi_max = get_max_bond_dim(l)
        right_chi_max = get_max_bond_dim(l+1)

        vprint("Determining Vidal-representation of MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)

        '''
        We define Q as a stacked set of canonical matrices: Q = [Q^{\sigma_1}, ..., Q^{\sigma_d}]^t
        '''
        if d*left_chi_max >= right_chi_max: # This condition is always fulfilled when the MPS is done from left to right.
            Q = random_unitary_matrix(d*left_chi_max)[:,:right_chi_max]
        else: # Called only when the MPS is done from right to left.
            Q = np.conj(random_unitary_matrix(right_chi_max)[:,:d*left_chi_max].T)

        # Check that the columns of Q are orthonormal.
        assert np.isclose(np.linalg.norm(np.eye(Q.shape[1]) - np.conj(Q.T)@Q), 0.0)

        '''
        Now we search V such that Id = Mfused^+ Mfused = V^+ Q^+ S^2 Q V where S^2 is the d-block-diagonal matrix with rho_l on its blocks and Mfused the desired fused set of site matrices
        In order to find V we set A = Q^+ S^2 Q, yielding V = A^{-1/2}
        '''
        V = solve_inversion(prev_S**2, Q)
        Mfused = Q @ V

        # Get next Schmidt-value distribution
        # next_S = np.ones(right_chi_max) / np.sqrt(right_chi_max) #random_eigenvalues_from_n_sphere(right_chi_max)
        next_S = truncate(np.abs(random_eigenvalues_from_n_sphere(right_chi_max)), threshold)

        rho = next_S**2
        assert np.isclose(np.sum(rho), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))

        list_rho.append(rho)

        # now decompose Mfused to get actual site tensors
        M_l = np.zeros([d, left_chi_max, right_chi_max], dtype =complex)

        S_matrix = np.reshape(np.diag(next_S), [len(next_S), len(next_S)])
        for idx in range(d):
            M_l[idx, :, :] = Mfused[idx*left_chi_max:(idx+1)*left_chi_max,:] @ S_matrix
            vprint("M_{}@site {}\n{}".format(idx, l, M_l[idx, :, :]), _verbosity, VLevel.DEBUG)
        list_M.append(np.copy(M_l))
        prev_S = next_S

    return list_M, list_rho


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
    vector_state = vector_state.reshape((d**L))
    state_flat = np.array([x[0] for x in vector_state])
    return state_flat.reshape(d**L)

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

def random_state_vector_from_mps(L: int, chi_max: int, d: int = 2)-> np.array:
    """
    Input:
    L: number of lattice sites
    chi_max: bond dimension?
    d: dimension of the local space

    Return:
    state: an array of the probability amplitude of a random state in the computational basis.
    """
    list_M, _ = produce_random_mps(L, chi_max, d, _verbosity = VLevel.QUIET)
    state = vector_state_from_mps(list_M, L, d)
    return state

def product_state_eigenvalues(n: int) ->np.array:
    eigenvalues = np.zeros(n)
    eigenvalues[0] = 1
    return eigenvalues

def generate_angle_product_state_eigenvalues(n: int)-> np.array:
    """
    Generate the spherical coordinates corresponding to the Cartesian coordinates given by 
    (1, 0,...,0).
    
    Return:
    phi_i (np.array of shape (n-1)): array of angles (in radian).
    """
    angles = np.ones(n-1)*np.pi/2 
    angles[0] = 0
    return angles

def maximally_entangled_eigenvalues(n: int)-> np.array:
    return np.ones(n)*(1/n)

def generate_angle_maximally_entangled_eigenvalues(n: int)-> np.array:
    """
    Generate the spherical coordinates corresponding to the Cartesian coordinates given by 
    (1/\sqrt{n},...,1/\sqrt{n}).
    
    Return:
    phi_i (np.array of shape (n-1)): array of angles (in radian).
    """
    phi_i = np.arccos([1/math.sqrt(n+1-i) for i in range(1,n)])
    return phi_i

def generate_angles_between_max_entangled_and_product_state_eigenvalues(n:int, t:float)-> np.array:
    """
    Generate the spherical coordinates on the n-sphere with r=1 corresponding to 
    the vector D(t) = v_0 + t (v_1 - v_0) with v_0, v_1 points on the n-sphere with 
    the following spherical coordinates: 
    v_0 = ( 0, pi/2,..., pi/2), the product state point.
    v_1 = np.arccos([1/math.sqrt(n+1-i) for i in range(1, n)]), the maximally mixed state.

    Input:
    t (float)
    n (int): the dimension of the n-sphere.

    Output:
    angles (np.array): array of angles (in radian) of dimension n-1.
    """
    if n==1:
        return np.array([])
    else: 
        v_0 = generate_angle_product_state_eigenvalues(n)
        v_1 = generate_angle_maximally_entangled_eigenvalues(n) 
        D = (1-t) * v_0 + t * v_1 
        return D

def random_point_on_the_n_sphere_in_a_disk(n: int, diameter_disk: float, angles_center: np.array, r: float=1):
    """
    Input:
    n (int): dimension of the space of the n-sphere. We should have angles_center.size = n-1. 
    diameter_disk (float): the diameter of the disk (in radian). We place the simulation in the positive orthant. 
                    Then, in order to keep a uniform distribution on the disk of the n-sphere we should have: 
                    epsilon_over_2 > math.asin(1/math.sqrt(n)), hence the ValueError line. 
    angles_center (np.array): coordinates of the point with is the center of the disk in which are taken 
                    the random point. By default, it will be the point giving the maximally mixed states. 
    # t (float): Float in [0,1]. In the case where we need to compute the center of the disk via the function 
    #             generate_angles_between_max_entangled_and_product_state_eigenvalues(n, t). It is the t quantifying 
    #             the distance between the center and the product state. 
    r (float): rayon of the n-sphere. By default, 1.

    Output: 
    new_coordinates (np.array): A random point uniformly distributed around the point whose coordinates are angles_center 
                    in a disk of diameter diameter_disk on the n-sphere. The point is given via its Cartesian 
                    coordinates. 
    """
    if angles_center.size != n-1:
        raise ValueError(f"The input array angles_center must be of size n-1.")
    rayon_disk = diameter_disk/2
    # if rayon_disk > min(math.asin(1/math.sqrt(n)), np.pi/4):
    #     raise ValueError(f"The input value diameter_disk must be smaller than {2*math.asin(1/math.sqrt(n))} and pi/2")
    # Generate the coordinates of the maximal
    if angles_center is None:
        print("Angles center is None")
        angles_center = generate_angle_maximally_entangled_eigenvalues(n)
    # Step 1: Generate the random angles
    delta_angles = np.random.uniform( -rayon_disk, rayon_disk, n - 1) # Δφ1, Δφ2, ..., Δφ_{n-1} in [-ε/2, +ε/2]
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

def random_point_on_the_n_sphere_in_an_annulus(n: int, int_diameter_disk: float, annulus_width: float , angles_center: np.array, r: float=1):
    """
    Input:
    n (int): dimension of the space of the n-sphere. We should have angles_center.size = n-1. 
    int_diameter_disk: r of the annulus
    annulus_width: R-r of the annulus
    angles_center (np.array): coordinates of the point with is the center of the disk in which are taken 
                the random point. By default, it will be the point giving the maximally mixed states. 

    Output: 
    new_coordinates (np.array): A random point uniformly distributed in the annulus of interior radius 
                    r=int_diameter_disk/2 and exterior radius R = r + annulus_width on the n-sphere.
                    The center of the annulus is the point whose coordinates are angles_center 
                    The point is given via its Cartesian coordinates. 
    """
    # Generate the coordinates of the maximal
    if angles_center is None:
        angles_center = generate_angle_maximally_entangled_eigenvalues(n)
    if angles_center.size != n-1:
        raise ValueError(f"The input array angles_center must be of size n-1.")
    int_rayon = int_diameter_disk/2
    # if rayon_disk > min(math.asin(1/math.sqrt(n)), np.pi/4):
    #     raise ValueError(f"The input value diameter_disk must be smaller than {2*math.asin(1/math.sqrt(n))} and pi/2")

    # Step 1: Generate the random angles
    position_in_annulus = np.random.uniform( -annulus_width, annulus_width, n - 1)
    delta_angles = position_in_annulus + int_rayon*position_in_annulus/abs(position_in_annulus)  
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



def produce_random_mps_eigenvalues_in_a_disk(L: int, chi_max: int, diameter_disk: float = 0.01, d: int = 2, _verbosity: int = VLevel.INFO):
    """
    Input:
    L (int): number of site (assumed even for simplicity for the moment).
    d (int): dimension of the hilbert space of a single site (for qubit d=2).
    chi_max: maximal bond dimension 

    Output:
    list_M (list): list of the tensors M_optimized ordered increasingly. 
    rho_middle
    
    The function constructs in the ascending order, the matrices M to produce a random product state. 
    """

    def get_max_bond_dim(l: int):
        '''
            l counts the bonds starting from the left
        '''
        return min(chi_max, np.power(d, l) if l <= L/2 else np.power(d, L-l))

    def truncate(_S, _threshold):

        S = np.copy(_S)
        Snorm = np.sum(S**2)
        S[S < _threshold] = 0.0

        return S * Snorm / np.sum(S**2)

    def solve_inversion(_D, _Q, _threshold):

        A = np.conj(Q.T) @ np.kron(np.eye(d), np.diag(_D)) @ Q

        vprint("Solving inverse square root of A = Q^+ S^2 Q, dim(A) = {} x {}".format(*A.shape), _verbosity, VLevel.DEBUG)
        U, S, Vh = recursive_svd(A, 1e-14, _verbosity = _verbosity)

        assert np.all(np.isclose(S.imag, 0.0)), "non-vanishing imaginary part in singular values detected, violated condition is np.isclose(S.imag, 0.0) = {}".format(np.isclose(S.imag, 0.0))

        S = truncate(S, _threshold)
        Sinv = np.array([1.0 / np.sqrt(s) if s > 0.0 else 0.0 for s in S])
        return np.conj(Vh.T) @ np.diag(Sinv) @ np.conj(U.T)


    threshold = 1.e-16
    prev_S = np.array([1.0])
    list_M = list()
    list_rho = list()

    for l in range(L):

        left_chi_max = get_max_bond_dim(l)
        right_chi_max = get_max_bond_dim(l+1)

        vprint("Determining Vidal-representation of MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)

        '''
        We define Mfused as the stacked set of MPS-matrices at site l: Mfused = [M^{\sigma_1}, ..., M^{\sigma_d}]^t := Q*V
        '''
        if d*left_chi_max >= right_chi_max:
            Q = random_unitary_matrix(d*left_chi_max)[:,:right_chi_max]
        else:
            Q = np.conj(random_unitary_matrix(right_chi_max)[:,:d*left_chi_max].T)

        assert np.isclose(np.linalg.norm(np.eye(Q.shape[1]) - np.conj(Q.T)@Q), 0.0)

        (col_dim, row_dim) = Q.shape
        '''
        Now Id = V^+ Q^+ S^2 Q V where S^2 is the d-block-diagonal matrix with rho_l on its blocks
        In order to find V we set A = Q^+ S^2 Q, yielding V = A^{-1/2}
        '''
        V = solve_inversion(prev_S**2, Q, threshold)
        Mfused = Q @ V

        # Get next Schmidt-value distribution
        angles_center_l = generate_angle_maximally_entangled_eigenvalues(right_chi_max)
        next_S = truncate(np.abs(random_point_on_the_n_sphere_in_a_disk(right_chi_max, diameter_disk, angles_center_l)), threshold)

        rho = next_S**2
        assert np.isclose(np.sum(rho), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))

        list_rho.append(rho)

        # now decompose Mfused to geht actual site tensors
        M_l = np.zeros([d, left_chi_max, right_chi_max], dtype =complex)

        S_matrix = np.reshape(np.diag(next_S), [len(next_S), len(next_S)])
        for idx in range(d):
            M_l[idx, :, :] = Mfused[idx*left_chi_max:(idx+1)*left_chi_max,:] @ S_matrix
            vprint("M_{}@site {}\n{}".format(idx, l, M_l[idx, :, :]), _verbosity, VLevel.DEBUG)

        list_M.append(np.copy(M_l))
        prev_S = next_S

    return list_M, list_rho


def produce_random_mps_eigenvalues_gaussian_in_a_disk(L: int, chi_max: int, standard_deviation: float = 0.01, d: int = 2, _verbosity: int = VLevel.INFO):
    """
    Input:
    L (int): number of site (assumed even for simplicity for the moment).
    d (int): dimension of the hilbert space of a single site (for qubit d=2).
    chi_max: maximal bond dimension 

    Output:
    list_M (list): list of the tensors M_optimized ordered increasingly. 
    rho_middle
    
    The function constructs in the ascending order, the matrices M to produce a random product state. 
    """

    def get_max_bond_dim(l: int):
        '''
            l counts the bonds starting from the left
        '''
        return min(chi_max, np.power(d, l) if l <= L/2 else np.power(d, L-l))

    def truncate(_S, _threshold):

        S = np.copy(_S)
        Snorm = np.sum(S**2)
        S[S < _threshold] = 0.0

        return S * Snorm / np.sum(S**2)

    def solve_inversion(_D, _Q, _threshold):

        A = np.conj(Q.T) @ np.kron(np.eye(d), np.diag(_D)) @ Q

        vprint("Solving inverse square root of A = Q^+ S^2 Q, dim(A) = {} x {}".format(*A.shape), _verbosity, VLevel.DEBUG)
        U, S, Vh = recursive_svd(A, 1e-14, _verbosity = _verbosity)

        assert np.all(np.isclose(S.imag, 0.0)), "non-vanishing imaginary part in singular values detected, violated condition is np.isclose(S.imag, 0.0) = {}".format(np.isclose(S.imag, 0.0))

        S = truncate(S, _threshold)
        Sinv = np.array([1.0 / np.sqrt(s) if s > 0.0 else 0.0 for s in S])
        return np.conj(Vh.T) @ np.diag(Sinv) @ np.conj(U.T)


    threshold = 1.e-16
    prev_S = np.array([1.0])
    list_M = list()
    list_rho = list()

    for l in range(L):

        left_chi_max = get_max_bond_dim(l)
        right_chi_max = get_max_bond_dim(l+1)

        vprint("Determining Vidal-representation of MPS tensor M@{}, dim(M_{}) = {} x {}".format(l, l, left_chi_max, right_chi_max), _verbosity, VLevel.INFO)

        '''
        We define Mfused as the stacked set of MPS-matrices at site l: Mfused = [M^{\sigma_1}, ..., M^{\sigma_d}]^t := Q*V
        '''
        if d*left_chi_max >= right_chi_max:
            Q = random_unitary_matrix(d*left_chi_max)[:,:right_chi_max]
        else:
            Q = np.conj(random_unitary_matrix(right_chi_max)[:,:d*left_chi_max].T)

        assert np.isclose(np.linalg.norm(np.eye(Q.shape[1]) - np.conj(Q.T)@Q), 0.0)

        (col_dim, row_dim) = Q.shape
        '''
        Now Id = V^+ Q^+ S^2 Q V where S^2 is the d-block-diagonal matrix with rho_l on its blocks
        In order to find V we set A = Q^+ S^2 Q, yielding V = A^{-1/2}
        '''
        V = solve_inversion(prev_S**2, Q, threshold)
        Mfused = Q @ V

        # Get next Schmidt-value distribution
        angles_center_l = generate_angle_maximally_entangled_eigenvalues(right_chi_max)
        next_S = truncate(np.abs(gaussian_point_on_the_n_sphere_in_a_disk(right_chi_max, standard_deviation, angles_center_l)), threshold)

        rho = next_S**2
        assert np.isclose(np.sum(rho), 1.0), "Normalization constraint violated for Schmidt value distribution, |np.sum(S**2)| = {}".format(np.sum(rho))

        list_rho.append(rho)

        # now decompose Mfused to geht actual site tensors
        M_l = np.zeros([d, left_chi_max, right_chi_max], dtype =complex)

        S_matrix = np.reshape(np.diag(next_S), [len(next_S), len(next_S)])
        for idx in range(d):
            M_l[idx, :, :] = Mfused[idx*left_chi_max:(idx+1)*left_chi_max,:] @ S_matrix
            vprint("M_{}@site {}\n{}".format(idx, l, M_l[idx, :, :]), _verbosity, VLevel.DEBUG)

        list_M.append(np.copy(M_l))
        prev_S = next_S

    return list_M, list_rho






def produce_a_random_mps_eigenvalues_in_a_disk(L, d:int=2, diameter_disk = 0.01):
    """
    Input:
    L (int): number of site (assumed even for simplicity for the moment).
    d (int): dimension of the hilbert space of a single site (for qubit d=2).
    
    Output:
    list_M (list): list of the tensors M_optimized ordered increasingly. 
    list_objective_error
    list_left_normalization_error
    rho_middle
    
    The function constructs in the decreasing order, the matrices M to produce a random product state. 
    """
    rho_l = np.array([[1]]) # Initial density matrix rho_L.
    list_M = [] # I store the matrices M in a list. 
    list_objective_error = []
    list_left_normalization_error = []
    
    for l in range(L, int(L/2), -1): # l> L/2
        d_L_minus_l_plus_1 = int(d**(L-l+1))
        angles_center_l = generate_angle_maximally_entangled_eigenvalues(d_L_minus_l_plus_1)
        s_i = np.square(random_point_on_the_n_sphere_in_a_disk(d_L_minus_l_plus_1, diameter_disk, angles_center_l))
        # Random unitary matrix (rho is a density matrix so Hermitian so diagonalizable by a unitary matrix).
        Q = random_unitary_matrix(d_L_minus_l_plus_1) 
        D = np.diag(s_i) # Uniform distribution, to be modified later. 
        rho_l_minus_one = Q @ D @ np.conj(Q).T

        # Initialize M randomly 
        M_initial = np.random.randn(d_L_minus_l_plus_1, d_L_minus_l_plus_1) + 1j * np.random.randn(d_L_minus_l_plus_1, d_L_minus_l_plus_1)
        M_initial = M_initial.flatten()
        # Constraints dictionary for scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint_M_after_middle, 'args': (d, l, L)}
        # Optimize M to minimize the loss while satisfying the constraint
        result = minimize(loss_function_M_after_middle, M_initial, args=(rho_l, rho_l_minus_one, d, l, L), 
                          constraints=constraints, method='SLSQP', options={'disp': False})
        # Put options={'disp': True} to have the information about the optimization. 
        # Reshape the result back to the original shape
        M_optimized = result.x.reshape(d_L_minus_l_plus_1, d_L_minus_l_plus_1)
        list_M.insert(0, reshape_array_after_middle(M_optimized, d, l))
        list_objective_error.insert(0,check_objective_after_middle(M_optimized, rho_l, rho_l_minus_one, d, l, L))
        list_left_normalization_error.insert(0, check_left_normalization_after_middle(M_optimized, d, l, L))
        rho_l = rho_l_minus_one
    rho_middle = rho_l

    for l in range(int(L/2), 0, -1): #l <= L/2
        d_l = int(d**(l))
        d_l_minus_one = int(d**(l-1))
        angles_center_l = generate_angle_maximally_entangled_eigenvalues(d_l_minus_one)
        s_i = np.square(random_point_on_the_n_sphere_in_a_disk(d_l_minus_one, diameter_disk, angles_center_l))
        # Random unitary matrix (rho is a density matrix so Hermitian so diagonalizable by a unitary matrix).
        Q = random_unitary_matrix(d_l_minus_one) 
        D = np.diag(s_i) # Uniform distribution, to be modified later. 
        rho_l_minus_one = Q @ D @ np.conj(Q).T
        
        # Find M minimising our objective function with the left normalization constraint. 
        M_initial = np.random.randn(d_l, d_l) + 1j * np.random.randn(d_l, d_l)
        M_initial = M_initial.flatten()
        # Constraints dictionary for scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint_M_before_middle, 'args': (d, l)}
        # Optimize M to minimize the loss while satisfying the constraint
        result = minimize(loss_function_M_before_middle, M_initial, args=(rho_l, rho_l_minus_one, d, l), 
                          constraints=constraints, method='SLSQP', options={'disp': False})
        # Reshape the result back to the original shape
        M_optimized = result.x.reshape(d_l, d_l)
        list_M.insert(0, reshape_array_before_middle(M_optimized, d, l))
        list_objective_error.insert(0, check_objective_before_middle(M_optimized, rho_l, rho_l_minus_one, d, l))
        list_left_normalization_error.insert(0, check_left_normalization_before_middle(M_optimized, d, l))
        rho_l = rho_l_minus_one

    return list_M, list_objective_error, list_left_normalization_error, rho_middle


def produce_a_random_mps_eigenvalues_in_a_disk_center_moving(L, d=2, t=1, diameter_disk = 0.01):
    """
    Input:
    L (int): number of site (assumed even for simplicity for the moment).
    d (int): dimension of the hilbert space of a single site (for qubit d=2).
    t (float): in [0,1]
    
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
    rayon_disk = diameter_disk/2    

    for l in range(L, int(L/2), -1): # l> L/2
        d_L_minus_l_plus_1 = int(d**(L-l+1))
        # if d_L_minus_l_plus_1>1:
            # minimum_value_rayon_l = min(t*math.acos(1/math.sqrt(d_L_minus_l_plus_1)), np.pi/2 - t*math.acos(1/math.sqrt(d_L_minus_l_plus_1)), t*math.asin(1/math.sqrt(d_L_minus_l_plus_1-1)),np.pi/4*(2-t))
            # if rayon_disk > minimum_value_rayon_l:
            #     raise ValueError(f"Input parameter diameter_disk must be smaller than {2*minimum_value_rayon_l}.")
        angles_center_l = generate_angles_between_max_entangled_and_product_state_eigenvalues(d_L_minus_l_plus_1, t)
        s_i = np.square(random_point_on_the_n_sphere_in_a_disk(d_L_minus_l_plus_1, diameter_disk, angles_center_l))
        # Random unitary matrix (rho is a density matrix so Hermitian so diagonalizable by a unitary matrix).
        Q = random_unitary_matrix(d_L_minus_l_plus_1) 
        D = np.diag(s_i) # Uniform distribution, to be modified later. 
        rho_l_minus_one = Q @ D @ np.conj(Q).T

        # Initialize M randomly 
        M_initial = np.random.randn(d_L_minus_l_plus_1, d_L_minus_l_plus_1) + 1j * np.random.randn(d_L_minus_l_plus_1, d_L_minus_l_plus_1)
        M_initial = M_initial.flatten()
        # Constraints dictionary for scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint_M_after_middle, 'args': (d, l, L)}
        # Optimize M to minimize the loss while satisfying the constraint
        result = minimize(loss_function_M_after_middle, M_initial, args=(rho_l, rho_l_minus_one, d, l, L), 
                          constraints=constraints, method='SLSQP', options={'disp': False})
        # Put options={'disp': True} to have the information about the optimization. 
        # Reshape the result back to the original shape
        M_optimized = result.x.reshape(d_L_minus_l_plus_1, d_L_minus_l_plus_1)
        list_M.insert(0, reshape_array_after_middle(M_optimized, d, l))
        list_objective_error.insert(0,check_objective_after_middle(M_optimized, rho_l, rho_l_minus_one, d, l, L))
        list_left_normalization_error.insert(0, check_left_normalization_after_middle(M_optimized, d, l, L))
        rho_l = rho_l_minus_one
    rho_middle = rho_l

    for l in range(int(L/2), 0, -1): #l <= L/2
        d_l = int(d**(l))
        d_l_minus_one = int(d**(l-1))
        # if d_l_minus_one > 1:
        #     minimum_value_rayon_l = min(t*math.acos(1/math.sqrt(d_l_minus_one)), np.pi/2 - t*math.acos(1/math.sqrt(d_l_minus_one)), t*math.asin(1/math.sqrt(d_l_minus_one-1)),np.pi/4*(2-t))
        #     if rayon_disk > minimum_value_rayon_l:
        #         raise ValueError(f"Input parameter diameter_disk must be smaller than {2*minimum_value_rayon_l}.")
        angles_center_l = generate_angles_between_max_entangled_and_product_state_eigenvalues(d_l_minus_one, t)
        s_i = np.square(random_point_on_the_n_sphere_in_a_disk(d_l_minus_one, diameter_disk, angles_center_l))
        # Random unitary matrix (rho is a density matrix so Hermitian so diagonalizable by a unitary matrix).
        Q = random_unitary_matrix(d_l_minus_one) 
        D = np.diag(s_i) # Uniform distribution, to be modified later. 
        rho_l_minus_one = Q @ D @ np.conj(Q).T
        
        # Find M minimising our objective function with the left normalization constraint. 
        M_initial = np.random.randn(d_l, d_l) + 1j * np.random.randn(d_l, d_l)
        M_initial = M_initial.flatten()
        # Constraints dictionary for scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint_M_before_middle, 'args': (d, l)}
        # Optimize M to minimize the loss while satisfying the constraint
        result = minimize(loss_function_M_before_middle, M_initial, args=(rho_l, rho_l_minus_one, d, l), 
                          constraints=constraints, method='SLSQP', options={'disp': False})
        # Reshape the result back to the original shape
        M_optimized = result.x.reshape(d_l, d_l)
        list_M.insert(0, reshape_array_before_middle(M_optimized, d, l))
        list_objective_error.insert(0, check_objective_before_middle(M_optimized, rho_l, rho_l_minus_one, d, l))
        list_left_normalization_error.insert(0, check_left_normalization_before_middle(M_optimized, d, l))
        rho_l = rho_l_minus_one

    return list_M, list_objective_error, list_left_normalization_error, rho_middle

def random_state_vector_from_mps_eigenvalues_in_a_disk(L, d=2, diameter_disk=0.01):
    list_M, _, _, _ = produce_a_random_mps_eigenvalues_in_a_disk(L, d, diameter_disk)
    state = vector_state_from_mps(list_M, L, d)
    return state

def random_state_vector_from_mps_eigenvalues_in_a_disk_center_moving(L: int, d: int=2, t: float=1, diameter_disk: float=0.01):
    list_M, _, _, _ = produce_a_random_mps_eigenvalues_in_a_disk_center_moving(L, d, t, diameter_disk)
    state = vector_state_from_mps(list_M, L, d)
    return state
