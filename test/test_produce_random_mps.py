import numpy as np
import scipy as sp
from distribution_functions.random_mps_sp import produce_random_mps, VLevel

L = 20
d = 2
chi_max = 32

# generate random MPS
list_M, list_rho = produce_random_mps(L, 32, d, VLevel.INFO)

# test eigenvalue distributions
T = np.array([[1.0]], dtype = complex)
for j in range(L):

    Mj = list_M[j]
    Mfused = np.concatenate((T @ Mj[0,:,:], T @ Mj[1,:,:]), axis = 0)
    Mrho = np.conj(Mfused.T) @ Mfused
    T = np.diag(np.sqrt(list_rho[j]))

    #print(np.vstack((list_rho[j], np.diag(Mrho))))
    #print(Mfused)
    #print(np.conj(Mfused.T) @ Mfused)

    #assert np.isclose(np.sum(np.diag(Mrho)), 1.0), "@site {}, np.sum(np.diag(Mrho)) = {}".format(j, np.sum(np.diag(Mrho)))
    print("@site {} distance |rho_j - S^2_j| = {}, normalization constraint |np.sum(np.diag(Mrho)) - 1.0| = {}".format(j, np.linalg.norm(list_rho[j] - np.diag(T)**2), np.sum(np.diag(Mrho)) - 1.0))

