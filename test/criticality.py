import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import iv  # modified Bessel function I_0

from distribution_functions.random_mps import (
    gaussian_point_on_the_n_sphere_in_a_disk,
    generate_angle_maximally_entangled_eigenvalues,
)

n = 100
n_simulations = 10000
eigvals_all = []
omega_max_all = []

angles_center = generate_angle_maximally_entangled_eigenvalues(n)
standard_deviation = 0.01

# Simulation loop
for sim in range(n_simulations):
    lambda_i = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center))
    lambda_i = np.sort(lambda_i)[::-1]
    eigvals_all.append(lambda_i)
    omega_max_all.append(np.max(lambda_i))

eigvals_all = np.array(eigvals_all)
omega_max_all = np.array(omega_max_all)

# Define empirical n(ω)
def n_empirical(omega, eigvals_all):
    return np.sum(eigvals_all > omega)

omega_vals = np.linspace(np.min(eigvals_all), np.max(eigvals_all), 500)
n_emp_vals = np.array([n_empirical(omega, eigvals_all) for omega in omega_vals]) / n_simulations

# Theoretical n(ω) for a given b and omega_max
def n_theoretical_single_sim(omega, b, omega_max):
    log_term = np.log(omega_max / omega)
    log_term = np.clip(log_term, 1e-10, None)  # Avoid log(0)
    arg = 2 * np.sqrt(b * log_term)
    return iv(0, arg)

# Compute mean theoretical n(ω) over simulations
def mean_n_theoretical(omega_vals, b)-> np.array:
    n_th_matrix = []
    for omega_max in omega_max_all:
        n_th = n_theoretical_single_sim(omega_vals, b, omega_max)
        n_th_matrix.append(n_th)
    return np.mean(n_th_matrix, axis=0)

# Loss function for fitting b
def loss_fn(b):
    n_th_mean = mean_n_theoretical(omega_vals, b[0])
    mask = n_th_mean < 100 # mask is a Boolean array of the same shape as n_th_mean.
    return np.sum((n_emp_vals[mask] - n_th_mean[mask])**2)

b_0 =-np.log(np.max(omega_max_all))
print(f"Initial guess for b_0=-log(w_max) = {b_0}.")
# Fit b
result = minimize(loss_fn, x0=[b_0], bounds=[(1e-6, None)])
b_opt = result.x[0]
print(f"Optimized b: {b_opt}")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
# Final theoretical curve
n_th_vals = mean_n_theoretical(omega_vals, b_opt)
mask = n_th_vals < 100
omega_vals_filtered = omega_vals[mask]
n_th_vals_filtered = n_th_vals[mask]

# Apply the same mask to the empirical values
n_emp_vals_filtered = n_emp_vals[mask]

# Avoid division by zero in relative error (clip to avoid 0 denominator)
eps = 1 # 1e-10
rel_error = np.abs(n_emp_vals_filtered - n_th_vals_filtered) / np.clip(n_emp_vals_filtered, eps, None)

# Compute statistics
mean_rel_error = np.mean(rel_error)
max_rel_error = np.max(rel_error)
print(f"Mean relative error: {mean_rel_error:.4f}")
print(f"Max relative error: {max_rel_error:.4f}")

# Plotting
plt.figure(figsize=(10,6))
plt.plot(omega_vals, n_emp_vals, label='Empirical n(ω)', color='blue')
plt.plot(omega_vals_filtered, n_th_vals_filtered, label='Theoretical n(ω) (fit b)', color='red', linestyle='--')
plt.xlabel('ω')
plt.ylabel('n(ω)')
plt.title('Mean number of eigenvalues > ω\n(Individual $\omega_{max}^{(sim)}$ per simulation)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"test_criticality_standard_deviation{standard_deviation}_fitted_b_variable_omega_max.png")
plt.show()
