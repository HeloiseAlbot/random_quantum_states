#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the domain n from 1 to 100
n = np.arange(1, 501)

# Compute arcsin(1 / sqrt(n))
y = np.arcsin(1 / np.sqrt(n))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(n, y, label=r'$\arcsin\left(\frac{1}{\sqrt{n}}\right)$', color='blue')
plt.xlabel('n')
plt.ylabel('Value')
plt.title(r'Plot of $\arcsin\left(\frac{1}{\sqrt{n}}\right)$ for $n \in [1, 500]$')

# Grid: major and minor
plt.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.8)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)


plt.legend()
plt.tight_layout()
plt.show()

# %%
