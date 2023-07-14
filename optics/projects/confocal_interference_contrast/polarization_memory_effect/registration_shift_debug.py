import numpy as np
import matplotlib.pyplot as plt
from projects.confocal_interference_contrast.get_memory_effect import MemoryEffect
from optics.utils.display import complex2rgb
from optics.utils import ft
from projects.confocal_interference_contrast.complex_phase_image import Interferogram
from projects.confocal_interference_contrast.polarization_memory_effect import log


def load_data(reference_path):
    """
        :return: a list with one image of each concentration (1 to 4).
        Each image has both polarization side by side (H and V)

        info :
            concentration = [1.  1.5 2.  2.5 3.  3.5 4. ]
            imgs_to_compare.shape = (7, 1700, 3840)
    """
    log.info(f'loading {reference_path}')
    data = np.load(reference_path)
    imgs_to_compare = data['imgs']
    concentrations = np.atleast_1d(data['concentrations'])
    return imgs_to_compare, concentrations


def compare(image_pair: np.ndarray, registration=None) -> np.ndarray:
    log.debug('Calculating interferograms by detecting the spectral peak in the Fourier space...')
    interferogram_pair = [Interferogram(image_pair[0], registration=registration)]
    if registration is None:
        registration = interferogram_pair[0].registration
        log.info(f'Spatial frequencies: {interferogram_pair[0].registration.shift} cycles/px, amplitudes: {interferogram_pair[0].registration.factor}.')
    interferogram_pair.append(Interferogram(image_pair[1], registration=registration))
    complex_pupil_image_pair = np.asarray([_.complex_object * _.amplitude for _ in interferogram_pair])
    log.debug(f'Spatial frequencies: {interferogram_pair[0].registration.shift} and {interferogram_pair[1].registration.shift} cycles/px, amplitudes: {interferogram_pair[0].registration.factor} and {interferogram_pair[1].registration.factor}.')

    log.debug('Registering the pupils based on the amplitude only...')
    abs_registration = ft.subpixel.register(np.abs(complex_pupil_image_pair[1]), np.abs(complex_pupil_image_pair[0]))
    log.debug(f'Shifting the images by {abs_registration.shift} px based on amplitude of complex pupils...')
    complex_pupil_image_pair[1] = ft.roll(complex_pupil_image_pair[1], -abs_registration.shift)

    log.debug('Minimizing the phase tilt difference (and complex amplitude ratio) by registering both complex pupil images in spatial frequency space...')
    interferogram_registration = ft.Reference(
        reference_data_ft=ft.ifftshift(complex_pupil_image_pair[0])).register(subject_ft=ft.ifftshift(complex_pupil_image_pair[1]))
    log.debug(f'interferogram tip-tilt correction registration = {interferogram_registration}')
    complex_pupil_image_pair[1] = ft.fftshift(interferogram_registration.image_ft)

    # log.debug(np.linalg.norm(complex_pupil_image_pair, axis=(-2, -1), keepdims=True))
    complex_pupil_image_pair /= np.linalg.norm(complex_pupil_image_pair, axis=(-2, -1), keepdims=True)

    return complex_pupil_image_pair, registration


if __name__ == "__main__":
    # reference_path = r"C:\Users\tvettenburg\Downloads\imgs_for_test.npz"  # 7 pairs
    reference_path = r"C:\Users\tvettenburg\Downloads\test_sample_2.5c_id_2023-05-01_18-55-36.npz"  # 5 pairs
    # reference_path = r"D:\scan_pol_memory\results\memory_effect_data\test_sample_2.5c_id_2023-05-01_18-55-36.npz"
    log.info(f'Loading data from {reference_path}...')
    imgs_to_compare, concentrations = load_data(reference_path)
    log.info(f'loaded data of shape {imgs_to_compare.shape} for concentrations {concentrations}.')

    imgs_to_compare = np.asarray(imgs_to_compare)
    # pol_m_effect = MemoryEffect(imgs_to_compare[10])
    registration = None  # or set to a specific Registration point in frequency space
    for experiment_index in range(len(imgs_to_compare)):
        side_by_side_image = imgs_to_compare[experiment_index]
        log.info(f'Testing data for concentration {concentrations[min(experiment_index, concentrations.size-1)]}...')

        log.debug(f'Splitting images of shape {side_by_side_image.shape}...')
        image_pair = np.asarray([side_by_side_image[..., :side_by_side_image.shape[-1] // 2],
                                 side_by_side_image[..., side_by_side_image.shape[-1] // 2:]], dtype=np.float32)
        log.debug(f'Split images to shape {image_pair.shape}.')

        complex_pupil_image_pair, registration = compare(image_pair, registration)

        log.debug('Displaying output...')
        difference = np.linalg.norm(np.diff(complex_pupil_image_pair, axis=0))
        normalized_difference = difference / np.linalg.norm(complex_pupil_image_pair[0])
        inner_product = np.sum(complex_pupil_image_pair[0] * complex_pupil_image_pair[1].conj())
        normalized_inner_product = inner_product / np.linalg.norm(complex_pupil_image_pair[0]) / np.linalg.norm(complex_pupil_image_pair[1])
        log.debug(f'|a|={np.linalg.norm(complex_pupil_image_pair[0])}, |b|={np.linalg.norm(complex_pupil_image_pair[1])}, <a|b>/(|a||b|)={normalized_inner_product}')
        log.info(f'|<a|b>|/(|a||b|) = {np.abs(normalized_inner_product):0.8f}, |b-a|/|a| = {normalized_difference:0.8f}')

    normalization = np.amax(np.abs(complex_pupil_image_pair)) / 2  # max == 2
    # pol_m_effect_coefficients, c_imgs = pol_m_effect.calculate_coefficients()

    complex_pupil_image_pair_ft = ft.fftshift(ft.fft2(complex_pupil_image_pair))
    normalization_ft = np.amax(np.abs(complex_pupil_image_pair_ft)) / 5.0

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    axs[0, 0].imshow(complex2rgb(complex_pupil_image_pair_ft[0] / normalization_ft))
    axs[0, 0].set_title('after reg H')
    axs[1, 0].imshow(complex2rgb(complex_pupil_image_pair_ft[1] / normalization_ft))
    axs[1, 0].set_title('after reg V')
    axs[0, 1].imshow(complex2rgb(complex_pupil_image_pair[0] / normalization))
    axs[0, 1].set_title('E after reg H')
    axs[1, 1].imshow(complex2rgb(complex_pupil_image_pair[1] / normalization))
    axs[1, 1].set_title('E after reg V')
    axs[0, 2].imshow(complex2rgb(complex_pupil_image_pair[0] * complex_pupil_image_pair[1].conj() / normalization ** 2))
    axs[0, 2].set_title("H.V'")
    axs[1, 2].imshow(complex2rgb((complex_pupil_image_pair[0] - complex_pupil_image_pair[1]) / normalization))
    axs[1, 2].set_title("H-V")

    log.info('Done.')
    plt.show(block=True)

