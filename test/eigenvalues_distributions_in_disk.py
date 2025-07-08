import numpy as np
import sys
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import linregress

from distribution_functions.random_mps import generate_angle_maximally_entangled_eigenvalues, random_point_on_the_n_sphere_in_a_disk

n = 1000
n_simulations = 1000
lambda_values = []
angles_center = generate_angle_maximally_entangled_eigenvalues(n)



##################### Plot of one sd #####################
# diameter_disk = 0.001
       

# for sim in range(n_simulations):
#     lambda_i = np.square(random_point_on_the_n_sphere_in_a_disk(n, diameter_disk, angles_center))
#     lambda_i = np.sort(lambda_i)[::-1]
#     lambda_values.append(lambda_i)  # Store for averaging

# # Compute the average over simulations
# lambda_avg = np.mean(lambda_values, axis=0)

# # Plot the results
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, n + 1), lambda_avg, marker='o', linestyle='-', label='Average $\lambda_i$')
# plt.xlabel("Index i")
# plt.ylabel(r"$\langle \lambda_i \rangle$")
# plt.yscale("log")  # Set logarithmic scale for lambda values
# plt.title(f"Average of sorted $\lambda_i$ over {n_simulations} Simulations")
# plt.legend()
# plt.grid()



# # # Save the plot in the same directory as this script
# plt.savefig(f"lambda_plot_in_disk_diameter_disk{diameter_disk}logscale.png")
# plt.show()





##################### Range of diameter_disks to test (logarithmically spaced) #####################
# diameter_disks = np.logspace(0, -3, num=20)  # From 1 to 0.001

# slopes = []
# slope_errors = []

# # Loop over diameter_disk
# for diameter_disk in diameter_disks:
#     lambda_values = []
    
#     for sim in range(n_simulations):
#         lambda_i = np.square(
#             random_point_on_the_n_sphere_in_a_disk(n, diameter_disk, angles_center)
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

# # Plot slope vs diameter_disk
# plt.figure(figsize=(8, 5))
# plt.errorbar(diameter_disks, slopes, yerr=slope_errors, fmt='o-', capsize=5, label="Slope $a$ with error")
# plt.xscale('log')
# plt.xlabel("diameter_disk  (log scale)")
# plt.ylabel("Slope $a$ of $\log(\lambda_i) = a \cdot i + b$")
# plt.title("Slope $a$ vs. diameter_disk with Error Bars")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# plt.savefig(f"sebastian_plot_diameter_disk_more_precised.png")
# plt.show()



##################### Spaced diameter_disks log #####################
# Generate list of diameters
diameter_disks = np.linspace(1, 0.001, num=50)

slopes = []
r_squared_vals = []

for diameter_disk in diameter_disks:
    lambda_values = []

    for sim in range(n_simulations):
        lambda_i = np.square(
            random_point_on_the_n_sphere_in_a_disk(n, diameter_disk, angles_center)
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
worst_diameter = diameter_disks[min_idx]

# Plot slope vs diameter_disk
plt.figure(figsize=(8, 6))
scatter = plt.scatter(diameter_disks, slopes, c=r_squared_vals, cmap='viridis', s=80, edgecolor='k')
plt.colorbar(scatter, label='$R^2$ (Goodness of Fit)')
plt.xlabel("diameter_disk")
plt.ylabel("Slope $a$ of $\log(\lambda_i) = a \cdot i + b$")
plt.title("Slope $a$ vs. diameter_disk (colored by $R^2$)")
plt.grid(True)

# Add annotation under plot
plt.figtext(0.5, -0.05,
            f"Minimum $R^2$ = {worst_r2:.4f} at diameter_disk = {worst_diameter:.4f}",
            wrap=True, horizontalalignment='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"sebastian_plot_diameter_disk_more_precised_n{n}.png", bbox_inches='tight')
plt.show()

