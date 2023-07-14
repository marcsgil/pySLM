import numpy as np
import matplotlib.pyplot as plt

from examples.instruments.slm import log
from optics.utils.display import complex2rgb, grid2extent
from optics.instruments.slm import PhaseSLM, DualHeadSLM
from optics.instruments.slm.meadowlark_pci_slm import MeadowlarkPCISLM


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2)
    ax_im = ax[0].imshow(np.zeros([400, 600]))
    ax[0].set_title('slm pattern')
    ax[1].set_title('1st order complex')

    # Define SLM using a figure window instead of a physical display
    slm = PhaseSLM(display_target=ax_im, deflection_frequency=[0.05, 0.1])
    # slm2 = MeadowlarkPCISLM(board_number=1, wavelength=633e-9, deflection_frequency=[0.05, 0.1])
    log.info(f'SLM of shape {slm.shape} initialized.')
    # Place something on the SLM
    slm_r = np.sqrt(sum(_ ** 2 for _ in slm.grid))
    slm_theta = np.arctan2(slm.grid[1], slm.grid[0])
    slm.modulate((slm_r < 150) * np.exp(1j * slm_theta))  # Show a vortex

    ax[1].imshow(complex2rgb(slm.complex_field), extent=grid2extent(slm.grid))

    plt.show(block=True)
