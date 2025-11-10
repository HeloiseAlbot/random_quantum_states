import numpy as np
import scipy as sp
from scipy.linalg import svd
from distribution_functions.random_mps import produce_random_mps, VLevel, vector_state_from_mps
from distribution_functions.property_calculation_function import partial_trace

L = 16
d = 2
chi_max = 32
# l = 4

# generate random MPS
list_M, list_rho = produce_random_mps(L, chi_max, d, VLevel.INFO)
#print(f"list_rho{list_rho}")
# test eigenvalue distributions
T = np.array([[1.0]], dtype = complex)
for j in range(L-1):

    Mj = list_M[j]
    Mfused = np.concatenate((Mj[0,:,:], Mj[1,:,:]), axis = 0)
    TMfused = np.concatenate((T @ Mj[0,:,:], T @ Mj[1,:,:]), axis = 0)
    T = np.conj(Mfused.T) @ TMfused

    print("@site {} ||D - (Mfused^*)•T•Mfused||: {}".format(j, sp.linalg.norm(np.diag(T) - list_rho[j])))

