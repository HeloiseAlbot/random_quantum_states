import numpy as np
import random
from scipy.stats import unitary_group, ortho_group, entropy  # Added entropy for KL divergence
from scipy.linalg import eigh
import scipy.linalg as la
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy.random import default_rng
from math import cos, sin
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D
import distribution.distribution_functions.random_mps_v2 as rmps
from scipy.stats import norm

# Parameters  
d = 2 
L = 4
n = int(d**(L)) # Total dimension of the state space
n_samples = 10000  # Number of random matrices/states to generate
chi_max = 4

# print("Generating random matrix product states (MPS)...")
# # Generate random matrices and collect their entries
# original_samples = np.zeros((n_samples, n), dtype=complex)
# for i in range(n_samples):
#     list_M, list_rho = rmps.draw_random_mps(L, chi_max, d, None)
#     original_samples[i], _ = rmps.vector_state_from_mps_v2(list_M, L, d)

# print("Applying a fixed unitary transformation to all configurations...")
# # Apply a fixed unitary transformation to all matrices
# U_fixed = unitary_group.rvs(n)
# transformed_samples = np.zeros((n_samples, n), dtype=complex)
# for i in range(n_samples):
#     transformed_samples[i] = np.dot(U_fixed, original_samples[i])

# print("Saving real parts of dataset to space-delimited text files...")
# # Save data tracking real elements across all iterations
# np.savetxt(f"original_samples_real_L{L}.txt", original_samples.real)
# np.savetxt(f"transformed_samples_real_L{L}.txt", transformed_samples.real)

original_samples = np.loadtxt(f"original_samples_real_L{L}.txt")
transformed_samples = np.loadtxt(f"transformed_samples_real_L{L}.txt")
print("Generating histograms and exporting to PDF plots...")

# Theoretical distribution parameters for Haar-random states
mean = 0.0
variance = 1.0 / (2 * n)  # Variance for real part: 1/(2n)
std_dev = np.sqrt(variance)

# # Loop over each entry index, plot, and output as separate publication-ready PDFs
bins = 50
for i in range(n):
    fig, ax = plt.subplots(figsize=(8, 5))
    original_data = original_samples[:, i]
    transformed_data = transformed_samples[:, i]
    
    # Plot empirical histogram and capture bin data to calculate errors
    counts, bin_edges, patches = ax.hist(
        original_data, bins=bins, density=True, alpha=0.5, color="blue", 
        label=r"$\sigma$-random matrices"
    )    
    ax.hist(transformed_data, bins=bins, density=True, alpha=0.5, color="orange", label="Transformed")
    
    # if i == 0:
    #     ax.set_title(f"Distribution of the {i+1}-st entry")
    # elif i == 1:
    #     ax.set_title(f"Distribution of the {i+1}-nd entry")
    # elif i == 2:
    #     ax.set_title(f"Distribution of the {i+1}-rd entry")
    # else:
    #     ax.set_title(f"Distribution of the {i+1}-th entry")
# Calculate the midpoints of each bin to evaluate the theoretical curve at those exact positions
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    theoretical_at_bins = norm.pdf(bin_centers, mean, std_dev)
    
    # --- ERROR QUANTIFICATION ---
    # 1. Mean Squared Error (MSE)
    mse = np.mean((counts - theoretical_at_bins) ** 2)
    
    # 2. Kullback-Leibler (KL) Divergence 
    # To avoid log(0) issues if a bin has 0 counts, add a tiny epsilon or filter out zeros
    epsilon = 1e-10
    p = counts + epsilon
    q = theoretical_at_bins + epsilon
    # Normalize to proper probability distributions for standard KL calculation
    p /= np.sum(p)
    q /= np.sum(q)
    kl_div = entropy(p, q)
    # -----------------------------

    # Plot the theoretical Gaussian distribution
    x = np.linspace(-0.7, 0.7, 1000)  # Range for the theoretical curve
    theoretical_pdf = norm.pdf(x, mean, std_dev)
    ax.plot(x, theoretical_pdf, color="red", linestyle="--", linewidth=2, label="Theoretical (Haar)")
    
    # Add error metrics cleanly to the plot labels or text box
    error_text = f"MSE: {mse:.5f}\nKL Div: {kl_div:.5f}"
    ax.text(0.05, 0.95, error_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(rf"Real part of $\langle e_{{{i}}} |\psi\rangle$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    
    # Export individual high-quality graphics files
    pdf_filename = f"entry_{i+1}_distribution_L{L}_with_theory4.pdf"
    plt.tight_layout()
    plt.savefig(pdf_filename, format="pdf")
    plt.close(fig)
    print(f"Saved: {pdf_filename} | MSE: {mse:.5f} | KL: {kl_div:.5f}")

print("Job executed successfully!")