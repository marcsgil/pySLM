import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join


if __name__ == '__main__':
    folder = r"./display_images/"
    image_names = [f for f in os.listdir(folder) if isfile(join(folder, f))]

    images = {}
    for name in image_names:
        images[name] = plt.imread(folder + name)

    display_names = ["original_", "psf_", "restored_"]
    nb_samples = 3

    fig, axs = plt.subplots(3, nb_samples, sharex=False, sharey=True)
    for idx_0, row_name in enumerate(["original", "psf multiplication", "restored"]):
        for idx_1 in range(nb_samples):
            axs[idx_0, idx_1].imshow(images[display_names[idx_0]+str((idx_1 + 1)) + r".png"])
        axs[idx_0, 0].set(ylabel=row_name)

    plt.show(block=True)

