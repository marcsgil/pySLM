import numpy as np
import logging

log = logging.getLogger(__name__)


def correction_from_pupil(measured_pupil_function: np.ndarray, attenuation_limit: float = 4) -> np.ndarray:
    """
    Calculates a function that compensates for a given measured aberration.
    In principle the correction should be 1 / measured_pupil_function so that
    ``1 / measured_pupil_function * measured_pupil_function = 1``
    however, a spatial light modulator cannot produce amplitudes larger than 1. The ideal return function is thus
    ``np.amin(measured_pupil_function) / measured_pupil_function * measure_pupil_function = np.amin(measured_pupil_function)``
    This can pose a problem when the amplitude of measured_pupil_function is too small. The attenuation_limit value thus
    limits the amplification at pixels that are measured to have an amplitude less than 1/attenuation_limit of its
    maximum value.

    :param measured_pupil_function: The measured aberration as a complex array with one value for each pixel in the
        spatial light modulator.
    :param attenuation_limit: A scalar number indicating how much amplitude reduction is allowed.

    :return: A complex array with complex argument that is opposite of that of measure_pupil_function, and amplitude
        that is, or approximates that of 1 / measured_pupil_function, but never lower than attenuation_limit.
        that is, or approximates that of 1 / measured_pupil_function, but never lower than attenuation_limit.
    """
    amplitude = np.abs(measured_pupil_function)
    amplitude = np.maximum(amplitude, np.finfo(float).eps)
    amplitude *= attenuation_limit / np.amax(amplitude)  # do not attenuate below the limit
    amplitude = np.maximum(amplitude, 1)  # clip amplifications
    amplitude /= np.amin(amplitude)  # do not attenuate more than necessary

    return np.exp(-1j * np.angle(measured_pupil_function)) / amplitude


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from optics.utils.display import complex2rgb

    # Input data:
    rng = np.linspace(-1, 1, 64)
    x = rng[:, np.newaxis]
    y = rng[np.newaxis, :]
    r2 = x**2 + y**2
    sigma = 0.5
    defocus = np.exp(2j * np.pi * 3 * r2) * np.exp(-0.5 * r2/(sigma**2))

    correction = correction_from_pupil(defocus)

    log.info(f"     pupil between {np.min(np.abs(defocus).ravel())} and {np.max(np.abs(defocus).ravel())}.")
    log.info(f"correction between {np.min(np.abs(correction).ravel())} and {np.max(np.abs(correction).ravel())}.")

    # plt.subplot(1, 2, 1)
    # plt.imshow(complex2rgb(defocus))
    # plt.title("pupil")
    #
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(complex2rgb(correction))
    # # plt.imshow(np.abs(correction))
    # # plt.colorbar()
    # plt.title("correction")
    #
    # plt.show()

