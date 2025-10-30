import numpy as np
import matplotlib.pyplot as plt
from distribution_functions.random_mps import (
    gaussian_point_on_the_n_sphere_in_a_disk,
    generate_angle_maximally_entangled_eigenvalues,
    random_point_on_the_n_sphere
)

# Parameters
d = 2  # local dimension, e.g., 2 for qubits
sigma_min = 1e1   # 10^1
sigma_max = 1e20  # 10^20
# sigma_range = range(sigma_min, sigma_max, 10) # l from 1 to 100
# logarithmic scale for sigma
n_points = 50
sigma_range = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_points)
l = 15
S_values = []
# standard_deviation = 0.001
n_samples = 20  # Number of samples to average over


# Functions to compute the standard derivation.
def standard_deviation_rupture_pt_power_law(n):
    """
    Power law. lambda x, a, b: a * x**(-b)
    R² = 0.9874 | Parameters: 0.754138, 0.540190 
    """
    a = 0.754138
    b = 0.540190 
    return a * n**(-b)

def standard_deviation_rupture_pt_power_law_plus_constant(n):
    """
    Power law. lambda x, a, b: a * x**(-b)
    R² = 0.9877 | Parameters: 0.663936, 0.499940, -0.003400 
    """
    a = 0.663936
    b = 0.499940
    c = -0.003400 
    return a * n**(-b)+c

def standard_deviation_rupture_pt_inverse_root(n):
    """
    Inverse root. lambda x, a, c: a / np.sqrt(x) + c
    R² = 0.9877 | Parameters: 0.664063, -0.003395 
    """
    a = 0.664063
    c = -0.003395 
    return a / np.sqrt(n) + c

def standard_deviation_rupture_pt_logarithmic_inverse(n):
    """
    Logarithmic inverse. lambda x, a, c: a / np.log(x) + c
    R² = 0.9849 | Parameters: 0.648471, -0.077584
    """
    a = 0.648471
    c = -0.077584
    return a / np.log(n) + c

# Functions to compute the von Neumann entropy. 
def S_von_neumann(eigvals, base=np.e):
    """Compute von Neumann entropy from eigenvalues."""
    eigvals = np.array(eigvals)
    eigvals = eigvals[eigvals > 0]  # avoid log(0)
    return -np.sum(eigvals * np.log(eigvals)) / np.log(base)


S_values = []
n = d ** l  # Dimension of the reduced density matrix
angles_center = generate_angle_maximally_entangled_eigenvalues(n)

for sigma in sigma_range:
    S_total = 0.0
    for _ in range(n_samples):
        eigvals = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, sigma, angles_center))  # returns n eigenvalues of reduced density matrix
        S_sample = S_von_neumann(eigvals, base=2)  # base 2 → entropy in bits
        S_total += S_sample
    S_avg = S_total / n_samples
    S_values.append(S_avg)


# Save entropy values
np.savez(
    f"von_neumann_entropy_vs_sigma_{n_samples}samples_l{l}_sigma_{sigma_min}to{sigma_max}.npz",
    entropy=S_values, l_values=l, std_values=sigma_range
)

# Plot
# plt.figure(figsize=(8, 5))
# plt.plot(sigma_range, S_values, 'o-', label=f'Average entanglement entropy ($n={n_samples}$ samples)')
# plt.xlabel('Standard deviation $\sigma$')
# plt.ylabel('Entanglement entropy $S(\sigma)$')
# # plt.title('Averaged entanglement entropy vs subsystem size')
# plt.legend()
# plt.tight_layout()
# plt.grid(True)
# # plt.savefig(f"entanglement_entropy_VS_size_for_standard_deviation_at_rupture_pt_power_law_plus_constant.png")
# plt.savefig(f"von_neumann_entropy_vs_sigma_{n_samples}samples_l{l}_sigma_{sigma_min}to{sigma_max}.png")
# plt.show()


# Plot
plt.figure(figsize=(8, 5))
plt.plot(sigma_range, S_values, 'o-', label=f'Average entanglement entropy ($n={n_samples}$ samples)')
plt.xscale('log')   # <-- log scale for sigma
plt.xlabel('Standard deviation $\sigma$ (log scale)')
plt.ylabel('Entanglement entropy $S(\sigma)$')
plt.legend()
plt.tight_layout()
plt.grid(True, which="both", ls="--")
plt.savefig(f"von_neumann_entropy_vs_sigma_{n_samples}samples_l{l}_sigma_{sigma_min}to{sigma_max}.png")
plt.show()
