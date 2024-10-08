# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Causality and Invertibility

## Causality

# %% import modules
import numpy as np

# %% create a polynomial object
p = np.polynomial.Polynomial([1, -7/6, 1/3])

# %% calculate the roots
p.roots()

# console ouput
# array([1.5, 2. ])

## Invertibility

# %% create polynomial object
p = np.polynomial.Polynomial([1, -5/2, 1])

# %% calculate the roots
p.roots()

# console ouput
# array([0.5, 2. ])
