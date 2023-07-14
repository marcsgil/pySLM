import time
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt

magnitude = 3/np.sqrt(3)
corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
uncorrected_modes = [17]
nb_modes = 18
num_samples = 15000

std_zernike_coeffs = np.zeros(nb_modes)
std_zernike_coeffs[corrected_modes] = magnitude
std_zernike_coeffs[uncorrected_modes] = 0.01 / np.sqrt(3)
rng = np.random.Generator(np.random.PCG64(seed=1))  # For the noise generation
coefficient_array = rng.normal(size=[num_samples, nb_modes]) * std_zernike_coeffs

plt.figure()
plt.plot(np.transpose(coefficient_array), 'o-', label="train_loss")
plt.title(f' magnitude_{magnitude:0.3f}_num_samples{num_samples}')
plt.show()
