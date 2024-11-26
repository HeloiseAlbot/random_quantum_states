import numpy as np
import matplotlib.pyplot as plt
from distribution_functions.property_calculation_function import average_bond_dimension_for_epsilon

# Parameters
num_states = 10  # Number of random quantum states
qubit_dim = 2      # Dimension of each qubit (2 for qubits)
num_qubits = 8   # Number of qubits
epsilon_values = np.logspace(-12, 0, 50)  # Range of epsilon values, logarithmically spaced

# Compute the average bond dimension for each epsilon
average_bond_dims = average_bond_dimension_for_epsilon(num_states, qubit_dim, num_qubits, epsilon_values)
average_bond_dims_truncating = average_bond_dimension_for_epsilon(num_states, qubit_dim, num_qubits, epsilon_values, truncating=True)
# np.save(f'average_bond_dims_nsamples{num_states}_nqubits{num_qubits}.npy', average_bond_dims)
# np.save(f'average_bond_dims_truncating_nsamples{num_states}_nqubits{num_qubits}.npy', average_bond_dims_truncating)
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, average_bond_dims, marker='o', linestyle='-', color='b', label='without truncation')
plt.plot(epsilon_values, average_bond_dims_truncating, marker='o', linestyle='-', color='r', label='truncating')
plt.xscale('log')
plt.xlabel('Truncation Threshold (epsilon)')
plt.ylabel('Average Bond Dimension')
plt.title('Average Bond Dimension vs. Truncation Threshold')
plt.grid(True)
plt.legend(loc='upper right')
# plt.savefig(f'average_bd_vs_truncation_threshold_nsamples{num_states}_nqubits{num_qubits}.png')
plt.show(block=False)