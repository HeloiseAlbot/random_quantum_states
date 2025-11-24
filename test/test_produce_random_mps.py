import numpy as np
import scipy as sp
from scipy.linalg import svd, eigh
from distribution_functions.random_mps import produce_random_mps_2, VLevel, vector_state_from_mps
from distribution_functions.property_calculation_function import partial_trace, partial_trace_2

from itertools import product

L = 4
d = 2
chi_max = 32
# l = 4

# generate random MPS
list_M, list_rho = produce_random_mps_2(L, 32, d, VLevel.INFO)
print(f"list_rho{list_rho}")

vector_state, sigma_tuples, labels_flat = vector_state_from_mps(list_M, L=L, d=d)
rho = np.outer(vector_state, np.conjugate(vector_state))
# dims (list): Dimensions of the subsystems.
# keep (list): Indices of subsystems to keep after tracing out.
dims = [2,2]
keep = [0]

print("sigma tuples: {}".format(sigma_tuples))
print("labales_flat: {}".format(labels_flat))

#rho_A = partial_trace( rho, dims, keep)
#U, S, Vh = svd(rho_A, lapack_driver = 'gesvd')
#print(f"list_rho[0]{list_rho[0]}")
#print(f"S{S}")

rho_A = partial_trace_2(d, L, rho, sigma_tuples, keep)
E, V = eigh(rho_A)
print(f"list_rho[0]{list_rho[0]}")
print(f"E{E}, sum(E)={np.sum(E)}")
