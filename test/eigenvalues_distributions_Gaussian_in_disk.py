import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from distribution_functions.random_mps import gaussian_point_on_the_n_sphere_in_a_disk, generate_angle_maximally_entangled_eigenvalues, random_point_on_the_n_sphere_in_a_disk

n = 100
n_simulations = 100
lambda_values = []
angles_center = generate_angle_maximally_entangled_eigenvalues(n)



##################### Plot of one sd #####################
standard_deviation = 0.001
       

for sim in range(n_simulations):
    lambda_i = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center))
    lambda_i = np.sort(lambda_i)[::-1]
    lambda_values.append(lambda_i)  # Store for averaging

# Compute the average over simulations
lambda_avg = np.mean(lambda_values, axis=0)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(range(1, n + 1), lambda_avg, marker='o', linestyle='-', label='Average $\lambda_i$')
plt.xlabel("Index i")
plt.ylabel("Average $\lambda_i$")
plt.yscale("log")  # Set logarithmic scale for lambda values
plt.title(f"Average of sorted $\lambda_i$ over {n_simulations} Simulations")
plt.legend()
plt.grid()



# # Save the plot in the same directory as this script
plt.savefig(f"lambda_plot_gaussian_standard_deviation{standard_deviation}logscale.png")
plt.show()





##################### Range of standard deviations to test (logarithmically spaced) #####################
# std_devs = np.logspace(0, -3, num=20)  # From 1 to 0.001

# slopes = []
# slope_errors = []

# # Loop over standard deviations
# for standard_deviation in std_devs:
#     lambda_values = []
    
#     for sim in range(n_simulations):
#         lambda_i = np.square(
#             gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center)
#         )
#         lambda_i = np.sort(lambda_i)[::-1]
#         lambda_values.append(lambda_i)

#     lambda_avg = np.mean(lambda_values, axis=0)

#     # Fit log(lambda_avg) = a * i + b
#     i_vals = np.arange(1, n + 1)
#     log_lambda = np.log(lambda_avg)

#     slope, intercept, r_value, p_value, std_err = linregress(i_vals, log_lambda)

#     slopes.append(slope)
#     slope_errors.append(std_err)

# # Plot slope vs standard deviation
# plt.figure(figsize=(8, 5))
# plt.errorbar(std_devs, slopes, yerr=slope_errors, fmt='o-', capsize=5, label="Slope $a$ with error")
# plt.xscale('log')
# plt.xlabel("Standard Deviation (log scale)")
# plt.ylabel("Slope $a$ of $\log(\lambda_i) = a \cdot i + b$")
# plt.title("Slope $a$ vs. Standard Deviation with Error Bars")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# plt.savefig(f"sebastian_plot.png")
# # # plt.show()



##################### Linearly spaced standard deviations #####################
# std_devs = np.linspace(0.2, 0.001, num=20)

# slopes = []
# r_squared_vals = []

# for standard_deviation in std_devs:
#     lambda_values = []
    
#     for sim in range(n_simulations):
#         lambda_i = np.square(
#             gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center)
#         )
#         lambda_i = np.sort(lambda_i)[::-1]
#         lambda_values.append(lambda_i)

#     lambda_avg = np.mean(lambda_values, axis=0)

#     i_vals = np.arange(1, n + 1)
#     log_lambda = np.log(lambda_avg)

#     slope, intercept, r_value, _, _ = linregress(i_vals, log_lambda)

#     slopes.append(slope)
#     r_squared_vals.append(r_value**2)  # Coefficient of determination

# # Plot slope vs standard deviation
# plt.figure(figsize=(8, 5))
# scatter = plt.scatter(std_devs, slopes, c=r_squared_vals, cmap='viridis', s=80, edgecolor='k')
# plt.colorbar(scatter, label='$R^2$ (Goodness of Fit)')
# plt.xlabel("Standard Deviation")
# plt.ylabel("Slope $a$ of $\log(\lambda_i) = a \cdot i + b$")
# plt.title("Slope $a$ vs. Standard Deviation (colored by $R^2$)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.savefig(f"sebastian_plot_more_precised.png")

