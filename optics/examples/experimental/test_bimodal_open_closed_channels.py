import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from optics import utils
from optics.utils.ft import Grid, MutableGrid
from optics.utils.display import complex2rgb, grid2extent

#
# A quick test of how the distribution of transmission values should look like when there are open channels present:
#
# https://arxiv.org/pdf/1304.5562.pdf


nb_modes = 1024
sigma = np.array([0.5, 1.0])
aspect_ratio = 1.00
nb_measured = int(0.49 * nb_modes)

rng = np.array(Grid(shape=nb_modes, extent=1.0, origin_at_center=False))
x_distances = np.mod(rng + 0.5, 1.0) - 0.5
x_distances = x_distances[:, np.newaxis] - x_distances[np.newaxis, :]
# x_distances = np.mod(rng[:, np.newaxis] - rng[np.newaxis, :] + 0.5, 1.0) - 0.5
# x_distances = np.mod(x_distances + 0.25, 0.5) - 0.25
other_side = np.logical_xor(rng[:, np.newaxis] < 0.0, rng[np.newaxis, :] < 0.0)
y_distances = other_side * aspect_ratio  # Light also has to traverse slab when points on opposing sides of slab
distances = np.sqrt((x_distances / sigma[0])**2 + (y_distances / sigma[1])**2)
correlations = np.exp(-0.5 * distances ** 2)

# correlations = np.exp(-0.5 * np.array(rng / sigma) ** 2)[np.newaxis, :] * np.ones((nb_modes, 1))
# for row in range(nb_modes):
#     correlations[row] = np.roll(correlations[row], row)

scattering_matrix = np.random.randn(nb_modes, nb_modes) + 1j * np.random.randn(nb_modes, nb_modes)
scattering_matrix *= correlations
# m = scipy.linalg.orth(m)
u, s, vh = scipy.linalg.svd(scattering_matrix)
# scattering_matrix = u @ np.diag(np.exp(2j * np.pi * np.random.rand(nb_modes))) @ vh
scattering_matrix = u @ vh

backscattering_singular_values = scipy.linalg.svdvals(scattering_matrix[:nb_measured, :nb_measured])
transmission_singular_values = scipy.linalg.svdvals(scattering_matrix[-nb_measured:, :nb_measured])

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(utils.display.complex2rgb(correlations, normalization=1))
axs[1, 0].imshow(utils.display.complex2rgb(scattering_matrix, normalization=10))
axs[0, 1].plot(transmission_singular_values)
axs[1, 1].hist(transmission_singular_values, bins=100)
# axs[0, 2].plot(backscattering_singular_values)
# axs[1, 2].hist(backscattering_singular_values, bins=100)

plt.show(block=True)
