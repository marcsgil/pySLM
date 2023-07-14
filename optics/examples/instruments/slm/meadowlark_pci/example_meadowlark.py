import numpy as np
import matplotlib.pyplot as plt

from examples.instruments.slm.meadowlark_pci import log
from optics.utils.display import complex2rgb, grid2extent
from optics.instruments.slm.meadowlark_pci_slm import MeadowlarkPCISLM

if __name__ == '__main__':
    log.info('Test starting...')

    with MeadowlarkPCISLM(wavelength=633e-9, deflection_frequency=[1/30, 1/30]) as slm:
        log.info(f'The SLM object is {slm} with shape {slm.shape}')

        log.info('Displaying vortex in first order.')
        slm.modulate(lambda x, y: (np.sqrt(x**2+y**2) < 500) * np.exp(1j * np.arctan2(y, x)))

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(complex2rgb(slm.complex_field), extent=grid2extent(slm.grid))
        axs[0].set(ylabel='x [px]', xlabel='y [px]', title='Hologram 1st order')
        axs[1].imshow(slm.image_on_slm, extent=grid2extent(slm.grid))
        axs[1].set(ylabel='x [px]', xlabel='y [px]', title='Image on the SLM')

        plt.show()

    log.info('Done.')
