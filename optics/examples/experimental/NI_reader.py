import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from optics.instruments.detector import PhotoDetector

inst_samples = 5000
nb_channels = 4
with PhotoDetector(nb_channels, nb_samples=inst_samples) as reader:
    run_for_seconds = 1

    data = np.zeros((nb_channels, inst_samples))
    time_scale = np.linspace(0, inst_samples * reader.dt, inst_samples)

    img = np.empty(nb_channels, dtype=object)
    fig, axs = plt.subplots(nb_channels)
    if np.shape(axs) == ():
        axs = [axs]
    for idx, ax in enumerate(axs):
        img[idx] = ax.plot(time_scale, data[idx])
        ax.set(xlabel='time period, s', ylabel='voltage response, V', title=f'Channel AI{idx}', ylim=[-0.3, 10])
    plt.show(block=False)

    start_t = timer()
    while timer() - start_t < run_for_seconds:
        try:  # Matplotlib might momentarily freeze the app, leading to an error from NI DAQ terminal
            data[:, :] = reader.read(inst_samples)  # np.random.random((4, inst_samples))
            for idx, im in enumerate(img):
                im[0].set_data(time_scale, data[idx] * -1)
            plt.pause(1e-10)
            plt.show(block=False)
        except Exception as e:
            print(e)

plt.show(block=True)


