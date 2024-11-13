#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Parameters 
epsilon = 0.001
# List of file names.
if epsilon == 0.001:
    file_names_entropy = [f"Histogram_entropy_vs_bd_entropy_nsamples10000000_nqubits10_epsilon{epsilon}_sim{i}.npy"
        for i in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    file_names_bd = [f"Histogram_entropy_vs_bd_bond_dimension_values_nsamples10000000_nqubits10_epsilon{epsilon}_sim{i}.npy"
        for i in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
else: # For epsilon = 0.001, the files with sim1 are empty.
    file_names_entropy = [
        f"Histogram_entropy_vs_bd_entropy_nsamples10000000_nqubits10_epsilon{epsilon}_sim{i}.npy"
        for i in range(11)]
    file_names_bd = [f"Histogram_entropy_vs_bd_bond_dimension_values_nsamples10000000_nqubits10_epsilon{epsilon}_sim{i}.npy"
        for i in range(11)]

# Initialize an empty list to hold the loaded arrays
arrays_entropy = []
arrays_bd = []

# Load each file and append the array to the list
for file_name in file_names_entropy:
    arrays_entropy.append(np.load(file_name))
for file_name in file_names_bd:
    arrays_bd.append(np.load(file_name))

# Concatenate all arrays along the first axis (default)

concatenated_array_entropy = np.concatenate(arrays_entropy, axis=0)
concatenated_array_bd = np.concatenate(arrays_bd, axis=0)

# Now, `concatenated_array` contains the data from all the files
print(concatenated_array_entropy.shape)
print(concatenated_array_bd.shape)

# Create the 2D histogram
# `bins` can be adjusted to control the resolution of the histogram
hist, x_edges, y_edges = np.histogram2d(concatenated_array_entropy, concatenated_array_bd, bins=50)

#%%
# Define the number of bins for entropy
entropy_bins = 50  # You can adjust this as needed

# Define the bins for bond dimensions as discrete values
bond_dimension_bins = np.arange(concatenated_array_bd.min(), concatenated_array_bd.max() + 1, 1)

# Create the 2D histogram
hist, x_edges, y_edges = np.histogram2d(concatenated_array_entropy, concatenated_array_bd, 
                                        bins=[entropy_bins, bond_dimension_bins])

# Plot the 2D histogram
plt.figure(figsize=(8, 6))
plt.imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
plt.colorbar(label='Joint Probability Density')

# Set labels and title
plt.xlabel('Entropy')
plt.ylabel('Bond Dimension')
plt.title(f'Distribution of Entropy vs Bond Dimension with epsilon={epsilon}')

# Set the y-ticks to match the bond dimensions
plt.yticks(np.arange(bond_dimension_bins.min(), bond_dimension_bins.max()))

# Show the plot
plt.show()

#%%
# Create a custom colormap where zero probability is white
cmap = plt.cm.viridis
cmap = mcolors.ListedColormap(['white'] + list(cmap.colors))
norm = mcolors.BoundaryNorm(boundaries=[-0.1, 0.1] + list(np.linspace(0.1, hist.max(), cmap.N - 1)), ncolors=cmap.N)

# Plot the 2D histogram
plt.figure(figsize=(8, 6))
plt.imshow(hist.T, origin='lower', aspect='auto', cmap=cmap, norm=norm,
           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
plt.colorbar(label='Joint Probability Density')

# Set labels and title
plt.xlabel('Entropy')
plt.ylabel('Bond Dimension')
plt.title(f'Distribution of Entropy vs Bond Dimension for epsilon={epsilon}')

# Set the y-ticks to match the bond dimensions
plt.yticks(np.arange(bond_dimension_bins.min(), bond_dimension_bins.max() + 1))

# Show the plot
plt.show()

# %%

# %%
