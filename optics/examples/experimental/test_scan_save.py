import numpy as np
import matplotlib.pyplot as plt


file = np.load('Scale_scan1.npy')

for idx in file:
    for image in idx:
        plt.imshow(image)
        plt.show(block=False)
        plt.pause(0.01)

plt.show()