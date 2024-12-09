import numpy as np
import sys

from distribution_functions.random_mps import random_state_vector_from_mps, random_state_vector_from_mps_eigenvalues_in_a_disk
from distribution_functions.different_distribution_random_states_and_matrices import Haar_measure_random_state
from distribution_functions.kl_divergence_tools import compute_kl_divergence_from_samples

# sys.argv = [name_of_the_program, diameter_disk]
print("sys.argv:{}".format(sys.argv))
diameter_disk = float(sys.argv[1])
# diameter_disk = 0.1
L = 4
d = 2

# The function 1 has the Haar measure distribution. 
args_dict_1 = {"N": d**L}
func1 = Haar_measure_random_state

# The function 1 is our distribution, the random mps state.
# args_dict_1 = {"L": L, "d": d}
# func1 = random_state_vector_from_mps

# # The function 2 is our distribution, the random mps state. 
# args_dict_2 = {"L": L, "d": d}
# func2 = random_state_vector_from_mps

# The function 2 is our distribution, the random mps state center around the maximally entangled state. 
args_dict_2 = {"L": L, "d": d, "diameter_disk": diameter_disk}
func2 = random_state_vector_from_mps_eigenvalues_in_a_disk

# args_dict_2 = {"N": d**L}
# func2 = Haar_measure_random_state

# Example usage
np.random.seed(42)

# Compute KL divergence
kl_result, samplesP, samplesQ = compute_kl_divergence(func1, func2, args_dict_1, args_dict_2, n_samples=10000, bins=100, range_proba=(-5, 5))
print("KL Divergence:", kl_result)

# Specify the file to save the value
output_numpy_file = f"kl_divergence_result_disk{diameter_disk}.npy"
output_numpy_file_samplesP = f"samplesP_Haar_measure_disk{diameter_disk}.npy"
output_numpy_file_samplesQ = f"samplesQ_Haar_measure_disk{diameter_disk}.npy"

np.save(output_numpy_file, kl_result) # Save KL result in a numpy file
np.save(output_numpy_file_samplesP, samplesP)
np.save(output_numpy_file_samplesQ, samplesQ)



