import numpy as np
import scipy as sp
import sys 
import json
from scipy.linalg import svd, eigh
# from distribution_functions.random_mps import vector_state_from_mps, produce_random_mps
from distribution_functions.random_mps_v2 import produce_random_mps_v2, VLevel, vector_state_from_mps_v2, draw_random_mps, iterative_solve_mps
from distribution_functions.property_calculation_function import partial_trace_2, partial_trace
from itertools import product

# sys.argv = [name_of_the_program, standard_deviation, L, chi_max]
# print("sys.argv:{}".format(sys.argv))
# standard_deviation = float(sys.argv[1])
standard_deviation = None
L = int(sys.argv[2])
chi_max = int(sys.argv[3])
number_tests = int(sys.argv[4])
d = 2
# number_tests = 1000
# therehold_acceptable_errors = 1e-6
list_last_errors = []
list_of_list_M = []
list_of_list_rho = []
list_of_list_errors = []
for idx_tests in range(number_tests):
        
    # generate random MPS
    try: 
        list_M, list_rho = draw_random_mps(L, chi_max, d, standard_deviation, VLevel.QUIET)
        list_M, list_errors = iterative_solve_mps(list_M, list_rho, _verbosity = VLevel.QUIET)
        # Store the data
        list_last_errors.append(list_errors[-1])
        list_of_list_M.append(list_M)
        list_of_list_rho.append(list_rho)
        list_of_list_errors.append(list_errors)
        print(f"list_errors[-1]:{list_errors[-1]}")
    except:
        pass
# count  = sum(x < therehold_acceptable_errors for x in list_last_errors)
# print(f"Number of state produced, rate:{len(list_last_errors)/number_tests}")
# print(f"count_under_threshold/number_tests:{count/len(list_last_errors)}")

# File of parameters
params = {
    "Lattice parameters": {
        "L": L,
        "d": d,
        "chi_max": chi_max,
    },
    "Random simulations parameters": {
        "Standard deviation": standard_deviation
    },
    "Simulation": {
        "Number of tests": number_tests
    }
}

# Save to params.json in a nicely formatted human-readable layout
with open(f"new_parameters_L{L}_standard_deviation{standard_deviation}_chi_max{chi_max}_number_tests{number_tests}.json", "w") as f:
    json.dump(params, f, indent=4)  # indent makes it human-readable

# np.savez(
#     f"simulation_data_L{L}_std{standard_deviation}_chi{chi_max}.npz",
#     list_M=list_of_list_M,
#     list_rho=list_of_list_rho,
#     list_errors=list_of_list_errors,
#     list_last_errors=list_last_errors)
# list_M_file = f"list_of_list_M_L{L}_standard_deviation{standard_deviation}_chi_max{chi_max}.npy"
# np.save(list_M_file, np.array(list_of_list_M, dtype=object), allow_pickle=True)

# list_rho_file = f"list_of_list_rho_L{L}_standard_deviation{standard_deviation}_chi_max{chi_max}.npy"
# np.save(list_rho_file, np.array(list_of_list_rho, dtype=object), allow_pickle=True)

list_errors_file = f"new_list_of_list_errors_L{L}_standard_deviation{standard_deviation}_chi_max{chi_max}number_tests{number_tests}.npy"
np.save(list_errors_file, np.array(list_of_list_errors, dtype=object), allow_pickle=True)

list_last_errors_file = f"new_list_last_errors_L{L}_standard_deviation{standard_deviation}_chi_max{chi_max}_number_tests{number_tests}.npy"
np.save(list_last_errors_file, np.array(list_last_errors, dtype=object), allow_pickle=True)

exit()
# print(f"list_M: {list_M}")
# print(f"list_rho:{list_rho}")
vector_state, sigma_tuples, labels_flat = vector_state_from_mps(list_M, L=L, d=d)
rho = np.outer(vector_state, np.conjugate(vector_state))
# keep (list): Indices of subsystems to keep after tracing out.

for idx in range(L-1):

    keep = np.arange(idx+1)
    rho_A = partial_trace_2(d, L, rho, sigma_tuples, keep)
    E, V = eigh(rho_A)
    E = np.sort(E)[::-1][:len(list_rho[idx])]

    print("rho_expected: {}, rho_actual: {}".format(list_rho[idx], E))
    print("||rho_expected - rho_actual|| = {}".format(np.linalg.norm(E - list_rho[idx])))
    print(f"E{E}, sum(E)={np.sum(E)}")

# test eigenvalue distributions
# T = np.array([[1.0]], dtype = complex)
# for j in range(L):

#     Mj = list_M[j]
#     Mfused = np.concatenate((T @ Mj[0,:,:], T @ Mj[1,:,:]), axis = 0)
#     Mrho = np.conj(Mfused.T) @ Mfused
#     T = np.diag(np.sqrt(list_rho[j]))

    #print(np.vstack((list_rho[j], np.diag(Mrho))))
    #print(Mfused)
    #print(np.conj(Mfused.T) @ Mfused)

    #assert np.isclose(np.sum(np.diag(Mrho)), 1.0), "@site {}, np.sum(np.diag(Mrho)) = {}".format(j, np.sum(np.diag(Mrho)))
    # print("@site {} normalization constraint |np.sum(np.diag(Mrho)) - 1.0| = {}".format(j, abs(np.sum(np.diag(Mrho)) - 1.0)))
    # print("@site {} distance |rho_j - S^2_j| = {}, normalization constraint |np.sum(np.diag(Mrho)) - 1.0| = {}".format(j, np.linalg.norm(list_rho[j] - np.diag(T)**2), np.sum(np.diag(Mrho)) - 1.0))



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
# # dims = [2,2] 
# # keep (list): Indices of subsystems to keep after tracing out.
# # keep = [0] 
# rho_A = partial_trace(rho, dims, keep)
# print(f"rho_A{rho_A}")
# E, V = eigh(rho_A)
# print(f"E:{E}")
# print(f"list_rho{list_rho}")
# print(f"E{E}, sum(E)={np.sum(E)}")
# Example: Bell state |Φ+> = (|00> + |11>)/√2
# phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
# rho = np.outer(phi_plus, phi_plus.conj())
# rho_0 = partial_trace(rho, dims, keep)
# print(rho_0)
