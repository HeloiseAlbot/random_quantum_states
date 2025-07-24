import numpy as np
import matplotlib.pyplot as plt
from distribution_functions.random_mps import (
    gaussian_point_on_the_n_sphere_in_a_disk,
    generate_angle_maximally_entangled_eigenvalues,
    random_point_on_the_n_sphere
)

# Parameters
d = 2  # local dimension, e.g., 2 for qubits
l_values = range(1, 31)  # l from 1 to 100
S_values = []
# standard_deviation = 0.001


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



for l in l_values:
    n = d ** l  # Hilbert space dimension of subsystem
    # standard_deviation = standard_deviation_rupture_pt_inverse_root(n)
    # angles_center = generate_angle_maximally_entangled_eigenvalues(n)
    # eigvals = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center))  # returns n eigenvalues of reduced density matrix
    eigvals = np.square(random_point_on_the_n_sphere(n, r=1))
    S = S_von_neumann(eigvals, base=2)  # base-2 for bits
    S_values.append(S)

# Save entropy values
np.save(f"S_values_for_standard_deviation_at_rupture_pt_power_law.npy", S_values)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(l_values, S_values, 'o-', label='Entanglement entropy $S(\ell)$')
plt.xlabel('Subsystem size $\ell$')
plt.ylabel('Entanglement entropy $S(\ell)$')
plt.title('Entanglement Entropy vs Subsystem Size')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(f"entanglement_entropy_VS_size_for_standard_deviation_at_rupture_pt_power_law_plus_constant.png")
plt.savefig(f"entanglement_entropy_VS_size_whole_positive_orthant.png")
plt.show()