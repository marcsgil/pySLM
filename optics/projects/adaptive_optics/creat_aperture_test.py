import numpy as np
from matplotlib import pyplot as plt

def create_aperture(Px, Py, x0, y0, r):
    # Create an array of x and y coordinates
    x = np.arange(Px)[:, np.newaxis]
    y = np.arange(Py)

    # Calculate the squared distance from each pixel to the central pixel
    distance_sq = (x - x0)**2 + (y - y0)**2

    # Create the aperture array
    aperture = np.where(distance_sq <= r**2, 1, 0)

    return aperture


aperture = create_aperture(1024, 1280, 500, 500, 300)
fig_z, axs_z = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(18, 8))
imz = axs_z.imshow(aperture)
axs_z.set(title=f" radius {300}")
fig_z.colorbar(imz)
plt.show()

