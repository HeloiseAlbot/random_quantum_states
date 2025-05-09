import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.special import iv  # modified Bessel function I_0


from distribution_functions.random_mps import gaussian_point_on_the_n_sphere_in_a_disk, generate_angle_maximally_entangled_eigenvalues, random_point_on_the_n_sphere_in_a_disk

n = 100
n_simulations = 10000
eigvals = []
angles_center = generate_angle_maximally_entangled_eigenvalues(n)


standard_deviation = 0.0815
# --- 1. Simulate the eigenvalues
for sim in range(n_simulations):
    lambda_i = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center))
    lambda_i = np.sort(lambda_i)[::-1]
    eigvals.append(lambda_i)  # Store for averaging

# --- 2. Compute omega_max
omega_max = np.max(eigvals)

# --- 3. Define theoretical n(omega)
def n_theoretical(omega, omega_max):
    b = -np.log(omega_max)
    log_term = np.log(omega_max / omega)
    arg = 2 * np.sqrt(b * log_term)
    return iv(0, arg)

# --- 4. Define empirical n(omega)
def n_empirical(omega, eigvals):
    return np.sum(eigvals > omega)

# --- 5. Omega range
omega_vals = np.linspace(np.min(eigvals), omega_max, 500)

# --- 6. Compute theoretical n(omega)

n_th_vals = n_theoretical(omega_vals, omega_max)

# --- 7. Mask theoretical values where n(omega) >= 100
mask = n_th_vals < 100
omega_vals_filtered = omega_vals[mask]
n_th_vals_filtered = n_th_vals[mask]

# --- 8. Compute empirical n(omega) for all omega (or only filtered if you prefer)
n_emp_vals = np.array([n_empirical(omega, eigvals) for omega in omega_vals])

# --- 9. Plot
plt.figure(figsize=(10,6))
plt.plot(omega_vals, n_emp_vals/n_simulations, label='Empirical n(ω)', color='blue')
plt.plot(omega_vals_filtered, n_th_vals_filtered, label='Theoretical n(ω) (n<100)', color='red', linestyle='--')
plt.xlabel('ω')
plt.ylabel('n(ω)')
plt.title('Mean number of eigenvalues > ω')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"test_criticality_standard_deviation{standard_deviation}.png")














##################### Plot of one sd #####################
standard_deviation = 0.01
       

