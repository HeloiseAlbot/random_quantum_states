import numpy as np
import sys

from distribution_functions.random_mps import random_point_on_the_n_sphere_in_a_disk, random_state_vector_from_mps, random_state_vector_from_mps_eigenvalues_in_a_disk, random_state_vector_from_mps_eigenvalues_in_a_disk_center_moving
from distribution_functions.different_distribution_random_states_and_matrices import Haar_measure_random_state
from distribution_functions.kl_divergence import compute_kl_divergence

# sys.argv = [name_of_the_program, diameter_disk]
print("sys.argv:{}".format(sys.argv))
diameter_disk = float(sys.argv[1])
t = float(sys.argv[2])
# diameter_disk = 0.1
# t = 0.5 
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
args_dict_2 = {"L": L, "d": d, "t": t, "diameter_disk": diameter_disk}
func2 = random_state_vector_from_mps_eigenvalues_in_a_disk_center_moving




# args_dict_2 = {"N": d**L}
# func2 = Haar_measure_random_state

# Example usage
np.random.seed(42)

# Compute KL divergence
kl_result = compute_kl_divergence(func1, func2, args_dict_1, args_dict_2, n_samples=10000, bins=100, range_proba=(-5, 5))
print("KL Divergence:", kl_result)

# Specify the file to save the value
output_file = f"kl_divergence_result_disk{diameter_disk}_t{t}.txt"

# Write the value to the file
with open(output_file, "w") as file:
    file.write(f"KL Divergence for diameter disk {diameter_disk} center such as t={t}: {kl_result}\n")

print(f"KL Divergence saved to {output_file}")