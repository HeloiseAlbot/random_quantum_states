import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from distribution_functions.random_mps import (
    gaussian_point_on_the_n_sphere_in_a_disk,
    generate_angle_maximally_entangled_eigenvalues,
)

d = 2
l_max = 20
l_values = np.arange(1, l_max)  # Subsystem sizes
std_values = np.linspace(0.01, 1.0, 30)  # Standard deviation values
n_samples = 20  # Number of samples to average

def S_von_neumann(eigvals, base=np.e):
    eigvals = np.array(eigvals)
    eigvals = eigvals[eigvals > 0]
    return -np.sum(eigvals * np.log(eigvals)) / np.log(base)

# Allocate entropy results
Z = np.zeros((len(l_values), len(std_values)))

# Compute average entropy
for i, l in enumerate(l_values):
    n = d ** l
    for j, std in enumerate(std_values):
        S_total = 0.0
        for _ in range(n_samples):
            angles_center = generate_angle_maximally_entangled_eigenvalues(n)
            eigvals = np.square(gaussian_point_on_the_n_sphere_in_a_disk(n, std, angles_center))
            S_total += S_von_neumann(eigvals)
        Z[i, j] = S_total / n_samples

np.savez(f"results_von_neumann_entropy_3D_surface_plot_average_over_{n_samples}_samples_until_{l_max-1}_gaussian_disk_log.npz", entropy=Z, l_values=l_values, std_values=std_values)


# Prepare meshgrid
L, STD = np.meshgrid(std_values, l_values)

# Plot
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(L, STD, Z, cmap='plasma', edgecolor='none')
fig.colorbar(surf, ax=ax, label=r"Avg. von Neumann Entropy $S$ ")

ax.set_xlabel("Standard Deviation")
ax.set_ylabel("Subsystem Size $\ell$")
ax.set_zlabel("Average Entropy $S$")
ax.set_title(f"3D Plot of von Neumann Entropy (Avg over {n_samples} samples)")

plt.tight_layout()
plt.savefig(f"von_neumann_entropy_3D_surface_plot_average_over_{n_samples}_samples_until_{l_max-1}_gaussian_disk_log.png", dpi=300, bbox_inches='tight')
plt.show()

