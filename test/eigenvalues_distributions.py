import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from distribution_functions.random_mps import random_eigenvalues_from_n_sphere

n = 100
n_simulations = 1000
lambda_values = []

for sim in range(n_simulations):
    lambda_i = np.square(random_eigenvalues_from_n_sphere(n, r=1))
    lambda_i = np.sort(lambda_i)[::-1]
    lambda_values.append(lambda_i)  # Store for averaging

# Compute the average over simulations
lambda_avg = np.mean(lambda_values, axis=0)

# Plot the results
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, n + 1), lambda_avg, marker='o', linestyle='-', label='Average $\lambda_i$')
# plt.xlabel("Index i")
# plt.ylabel("Average $\lambda_i$")
# plt.yscale("log")  # Set logarithmic scale for lambda values
# plt.title(f"Average of sorted $\lambda_i$ over {n_simulations} Simulations")
# plt.legend()
# plt.grid()

# Optional: set Seaborn style
sns.set(style="whitegrid", context="talk", font_scale=1.2)

# Perform linear regression in log-log space (since y-axis is log scale)
x = np.arange(1, n + 1)
y = np.log(lambda_avg)  # log of y because the plot uses log scale

slope, intercept, r_value, p_value, std_err = linregress(x, y)
y_fit = np.exp(intercept + slope * x)  # back-transform to linear scale


# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot
ax.plot(
    range(1, n + 1), 
    lambda_avg, 
    marker='.', 
    linestyle='-', 
    color='tab:blue',
    label=r'$\langle \lambda_i \rangle$')

# Plot regression line in red
ax.plot(
    x, 
    y_fit, 
    color='red', 
    linestyle='--',
    linewidth=2,
    label=r"Linear regression ($y=a*i+b$)")
# Log scale for y-axis
ax.set_yscale("log")
# Labels and title
ax.set_xlabel("Index $i$", fontsize=20)
ax.set_ylabel(r"$\langle \lambda_i \rangle$", fontsize=20)
# ax.set_title(f"Average of sorted eigenvalues $\lambda_i$ over {n_simulations} simulations", fontsize=15)
# Ticks
ax.tick_params(axis='both', which='major', labelsize=16)
# Legend
ax.legend(fontsize=12)
# Layout and grid
fig.subplots_adjust(left=0.14, right=0.95, top=0.92, bottom=0.15)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# ax.tick_params(axis='both', which='both',  length=6, width=1)
# ax.tick_params(top=True, right=True)
plt.show()


# Save the plot in the same directory as this script
plt.savefig("lambda_plot6_logscale_reglin.png")
plt.show()
