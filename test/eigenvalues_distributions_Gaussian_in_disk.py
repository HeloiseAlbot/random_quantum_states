import numpy as np
import sys
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import linregress
from distribution_functions.random_mps import gaussian_point_on_the_n_sphere_in_a_disk, generate_angle_maximally_entangled_eigenvalues, random_point_on_the_n_sphere_in_a_disk

n = 64
sigma_limit = 1
n_simulations = 1000
lambda_values = []
angles_center = generate_angle_maximally_entangled_eigenvalues(n)



##################### Plot of one sd #####################
# standard_deviation = 0.3
       

# for sim in range(n_simulations):
#     lambda_i = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center))
#     lambda_i = np.sort(lambda_i)[::-1]
#     lambda_values.append(lambda_i)  # Store for averaging

# # Compute the average over simulations
# lambda_avg = np.mean(lambda_values, axis=0)

# # Prepare x and log(y) for regression
# x = np.arange(1, n + 1)
# log_lambda_avg = np.log(lambda_avg)

# # Perform linear regression on log(lambda_avg)
# slope, intercept, r_value, p_value, std_err = linregress(x, log_lambda_avg)

# # Compute the fitted values in original scale
# lambda_fit = np.exp(intercept + slope * x)

# # Plot the results
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, n + 1), lambda_avg, marker='o', linestyle='-', label=r"$\langle \lambda_i \rangle$")
# plt.plot(x, lambda_fit, color='red', linestyle='--', linewidth=2, label=f'Linear regression ($y={slope:.3f}*i+{intercept:.3f}$)')
# plt.xlabel("Index i")
# plt.ylabel(r"$\langle \lambda_i \rangle$")
# plt.yscale("log")  # Set logarithmic scale for lambda values
# plt.title(r"Average values of the sorted eigenvalues: Gaussian distribution with $\sigma=0.3$", fontsize=13)
# plt.legend()
# plt.grid()



# # # Save the plot in the same directory as this script
# plt.savefig(f"lambda_plot_gaussian_n{n}_standard_deviation{standard_deviation}logscale_reglin_with_truncationchanging_corrected.png")
# plt.show()





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


# plt.savefig(f"linear_fit_n{n}_standard_dev.png")
# # plt.show()



##################### Linearly spaced standard deviations #####################
std_devs = np.linspace(sigma_limit, 0.5, num=100)

slopes = []
r_squared_vals = []

for standard_deviation in std_devs:
    lambda_values = []
    
    for sim in range(n_simulations):
        lambda_i = np.square(
            gaussian_point_on_the_n_sphere_in_a_disk(n, standard_deviation, angles_center)
        )
        lambda_i = np.sort(lambda_i)[::-1]
        lambda_values.append(lambda_i)

    lambda_avg = np.mean(lambda_values, axis=0)

    i_vals = np.arange(1, n + 1)
    log_lambda = np.log(lambda_avg)

    slope, intercept, r_value, _, _ = linregress(i_vals, log_lambda)

    slopes.append(slope)
    r_squared_vals.append(r_value**2)  # Coefficient of determination


# Find min R² and corresponding diameter
r_squared_vals = np.array(r_squared_vals)
min_idx = np.argmin(r_squared_vals)
worst_r2 = r_squared_vals[min_idx]
worst_diameter = std_devs[min_idx]

# Plot slope vs standard deviation
plt.figure(figsize=(8, 5))
scatter = plt.scatter(std_devs, slopes, c=r_squared_vals, cmap='viridis', s=80, edgecolor='k')
plt.colorbar(scatter, label='$R^2$ (Goodness of Fit)')
plt.xlabel("Standard Deviation")
plt.ylabel("Slope $a$ of $\log(\lambda_i) = a \cdot i + b$")
plt.title("Slope $a$ vs. Standard Deviation (colored by $R^2$)")
plt.grid(True)

# Add annotation under plot
# plt.figtext(0.5, -0.05,
#             f"Minimum $R^2$ = {worst_r2:.4f} for the standard deviation = {worst_diameter:.4f}",
#             wrap=True, horizontalalignment='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"sebastian_plot_more_precised_n{n}_sigmalimit_{sigma_limit}_withtrucationchanging_corrected.png", bbox_inches='tight')
plt.show()
print("Minimum $R^2$ = {worst_r2} for the standard deviation = {worst_diameter}")


