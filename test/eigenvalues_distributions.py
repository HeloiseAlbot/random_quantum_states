import numpy as np
import sys
import matplotlib.pyplot as plt

from distribution_functions.random_mps import random_eigenvalues_from_n_sphere

n = 100
n_simulations = 500
lambda_values = []

for sim in range(n_simulations):
    lambda_i = np.square(random_eigenvalues_from_n_sphere(n, r=1))
    # lambda_i_sorted = np.sort(lambda_i)[::-1]  # They are already sorted in decreasing order in the function random_eigenvalues_from_n_sphere 
    lambda_values.append(lambda_i_sorted)  # Store for averaging

# Compute the average over simulations
lambda_avg = np.mean(lambda_values, axis=0)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(range(1, n + 1), lambda_avg, marker='o', linestyle='-', label='Average $\lambda_i$')
plt.xlabel("Index i")
plt.ylabel("Average $\lambda_i$")
plt.yscale("log")  # Set logarithmic scale for lambda values
plt.title(f"Average of Sorted $\lambda_i$ over {n_simulations} Simulations")
plt.legend()
plt.grid()
plt.show()

# Save the plot in the same directory as this script
plt.savefig("lambda_plot.png")
plt.show()
