import numpy as np
import sys
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import linregress

from distribution_functions.random_mps import generate_angle_maximally_entangled_eigenvalues, random_point_on_the_n_sphere_in_an_annulus


n_start = 10
n_stop = 200
n_increment = 10

n_simulations = 1000
worst_r2_list = []
worst_diameter_list = []
n_list = []

R_minus_r = 0.001

for n in range(n_start, n_stop, n_increment):
    lambda_values = []
    angles_center = generate_angle_maximally_entangled_eigenvalues(n)
    ##################### Linearly spaced standard deviations #####################
    limit_orthant = np.arcsin(1/np.sqrt(n))
    int_diam_limit = limit_orthant - R_minus_r
    int_diameter_disks = np.linspace(0, int_diam_limit, num=100)
    slopes = []
    r_squared_vals = []
    for int_diam in int_diameter_disks:
        lambda_values = []
        
        for sim in range(n_simulations):
            lambda_i = np.square(
                random_point_on_the_n_sphere_in_an_annulus(n, int_diam, R_minus_r, angles_center)
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
    worst_diameter = int_diameter_disks[min_idx]

    # Save the data in the lists.
    worst_r2_list.append(worst_r2)
    worst_diameter_list.append(worst_diameter)
    n_list.append(n)

n_array = np.array(n_list)
worst_diameter_array = np.array(worst_diameter_list)
worst_r2_array = np.array(worst_r2_list)

np.savez(f'rupture_point_from_{n_start}_to_{n_stop}_n_simulations{n_simulations}_annulis_width_band{R_minus_r}.npz', n_array=n_array, worst_diameter_array=worst_diameter_array, worst_r2_array=worst_r2_array)



