import numpy as np

from distribution_functions.random_mps import random_state_vector_from_mps
from distribution_functions.different_distribution_random_states_and_matrices import Haar_measure_random_state
from distribution_functions.kl_divergence import compute_kl_divergence

L = 4
d = 2

# The function 1 has the Haar measure distribution. 
args_dict_1 = {"N": d**L}
func1 = Haar_measure_random_state

# The function 2 is our distribution. 
args_dict_2 = {"L": L, "d": d}
func2 = random_state_vector_from_mps

# args_dict_2 = {"N": d**L}
# func2 = Haar_measure_random_state

# Example usage
np.random.seed(42)

# Compute KL divergence
kl_result = compute_kl_divergence(func1, func2, args_dict_1, args_dict_2, n_samples=10, bins=100, range_proba=(-5, 5))
print("KL Divergence:", kl_result)
