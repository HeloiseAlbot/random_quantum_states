import numpy as np
import scipy as sp
from distribution_functions.random_mps import produce_random_mps, VLevel
from itertools import product

L = 10
d = 2
chi_max = 32
l=4

# generate random MPS
list_M, list_rho = produce_random_mps(L, 32, d, VLevel.INFO)

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



def check_mps_relation(list_M, list_Lambda, l, d):
    """
    Check ∑_{σ1...σl} (M^{σ_l})†...(M^{σ_1})† M^{σ_1}...M^{σ_l} = Λ^{l}
    with Λ taken from list_Lambda[l].
    """
    # --- Use Λ ---
    Lambda = np.diag(list_Lambda[l-1])
    RHS = Lambda

    # --- Initialize LHS dynamically ---
    LHS = np.zeros_like(RHS, dtype=complex)

    for sigma_seq in product(range(d), repeat=l):
        prod_M = None
        for site_index, s in enumerate(sigma_seq):
            M_site = list_M[site_index]
            M_sigma = M_site[s]
            prod_M = M_sigma if prod_M is None else prod_M @ M_sigma

        a = prod_M.conj().T @ prod_M

        # Adapt size if first time
        if LHS.shape != a.shape:
            LHS = np.zeros_like(a)
        LHS += a

    diff_norm = np.linalg.norm(LHS - RHS)
    print(f"‖LHS - Λ^{l}‖ = {diff_norm:.3e}")
    if diff_norm < 1e-10:
        print("Relation holds numerically.")
    else:
        print("Relation deviates.")

    return LHS, RHS, Lambda

LHS, RHS, Lambda = check_mps_relation(list_M, list_rho, l, d)