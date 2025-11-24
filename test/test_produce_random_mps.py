import numpy as np
import scipy as sp
from scipy.linalg import svd, eigh
from distribution_functions.random_mps import vector_state_from_mps
from distribution_functions.random_mps_v2 import produce_random_mps_v2, VLevel

from distribution_functions.property_calculation_function import partial_trace, partial_trace_2

from itertools import product

L = 10
d = 2
chi_max = 2**L
# l = 4

# generate random MPS
list_M, list_rho = produce_random_mps_v2(L, 32, d, VLevel.INFO)
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

# list_M, list_rho = produce_random_mps_v2(L, 32, d, VLevel.INFO)
# print(f"list_M:{list_M}")
# # print(f"list_M[-1]:{list_M[-1]}")
# print(f"list_rho{list_rho}")
# # test eigenvalue distributions
# T = np.array([[1.0]], dtype = complex)
# for j in range(L):

#     Mj = list_M[j]
#     Mfused = np.concatenate((T @ Mj[0,:,:], T @ Mj[1,:,:]), axis = 0)
#     Mrho = np.conj(Mfused.T) @ Mfused
#     T = np.diag(np.sqrt(list_rho[j]))

#     #print(np.vstack((list_rho[j], np.diag(Mrho))))
#     #print(Mfused)
#     #print(np.conj(Mfused.T) @ Mfused)

#     #assert np.isclose(np.sum(np.diag(Mrho)), 1.0), "@site {}, np.sum(np.diag(Mrho)) = {}".format(j, np.sum(np.diag(Mrho)))
#     print("@site {} normalization constraint |np.sum(np.diag(Mrho)) - 1.0| = {}".format(j, abs(np.sum(np.diag(Mrho)) - 1.0)))
#     # print("@site {} distance |rho_j - S^2_j| = {}, normalization constraint |np.sum(np.diag(Mrho)) - 1.0| = {}".format(j, np.linalg.norm(list_rho[j] - np.diag(T)**2), np.sum(np.diag(Mrho)) - 1.0))



# def check_mps_relation(list_M, list_Lambda, l, d):
#     """
#     Check ∑_{σ1...σl} (M^{σ_l})†...(M^{σ_1})† M^{σ_1}...M^{σ_l} = Λ^{l}
#     with Λ taken from list_Lambda[l].
#     """
#     # --- Use Λ ---
#     Lambda = np.diag(list_Lambda[l-1])
#     RHS = Lambda

#     # --- Initialize LHS dynamically ---
#     LHS = np.zeros_like(RHS, dtype=complex)

#     for sigma_seq in product(range(d), repeat=l):
#         prod_M = None
#         for site_index, s in enumerate(sigma_seq):
#             M_site = list_M[site_index]
#             M_sigma = M_site[s]
#             prod_M = M_sigma if prod_M is None else prod_M @ M_sigma

#         a = prod_M.conj().T @ prod_M

#         # Adapt size if first time
#         if LHS.shape != a.shape:
#             LHS = np.zeros_like(a)
#         LHS += a

#     diff_norm = np.linalg.norm(LHS - RHS)
#     print(f"‖LHS - Λ^{l}‖ = {diff_norm:.3e}")
#     if diff_norm < 1e-10:
#         print("Relation holds numerically.")
#     else:
#         print("Relation deviates.")

#     return LHS, RHS, Lambda

# LHS, RHS, Lambda = check_mps_relation(list_M, list_rho, l, d)

# vector_state = vector_state_from_mps(list_M, L=L, d=d)

# rho = np.outer(vector_state, np.conjugate(vector_state))
# print(f"rho{rho}")
# # dims (list): Dimensions of the subsystems. For L=3, one can use dims = [4,2] or dims = [2,4]. 
# dims = [2,2] 
# # keep (list): Indices of subsystems to keep after tracing out.
# keep = [0] 
# rho_A = partial_trace(rho, dims, keep)
# print(f"rho_A{rho_A}")
# # Example: Bell state |Φ+> = (|00> + |11>)/√2
# # phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
# # rho = np.outer(phi_plus, phi_plus.conj())
# # rho_0 = partial_trace(rho, dims, keep)
# # print(rho_0)


# U, S, Vh = svd(rho_A, lapack_driver = 'gesvd')
# print(f"list_rho[0]{list_rho[0]}")
# print(f"S{S}")



# def partial_trace_rho_qubit0(rho):
#     """
#     Compute the reduced density matrix of qubit 0
#     by tracing out qubit 1 from a 2-qubit density matrix (4x4).
#     """
#     rho = np.asarray(rho)
#     if rho.shape != (4, 4):
#         raise ValueError("Input density matrix must be 4x4 (2 qubits).")

#     # Reshape to 2x2x2x2
#     rho_reshaped = rho.reshape(2, 2, 2, 2)
#     # Trace out qubit 1 (the second qubit)
#     rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
#     return rho_reduced
