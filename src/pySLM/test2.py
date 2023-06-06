import numpy as np
from initial_profiles import *

xi = np.linspace(-1, 1,10)

x, y = np.meshgrid(xi, xi)

print(tilted_lens(x,y, 1, 1, 1064))