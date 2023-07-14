"""
Simulate A-scan of discrete reflectors, approximate the 400kHz swept source
based on Dorian Urban's Matlab code
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.constants as const
from scipy.signal.windows import hann
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

from optics.utils import ft
from projects.oct import log, normxcorr2


def tfridge(frequency_time_array, frequencies, frequency_change_penalty: float = 1e-2):
    """This is a heavily simplified version of the real thing!"""
    f_index = np.argmax(frequency_time_array, axis=0)
    return frequencies[f_index]


def smooth_interp(xi, x, y):
    result = Akima1DInterpolator(x, y)(xi)

    if np.any(np.isnan(result)):
        raise ValueError(f'Extrapolation resulted in NaN: {result}')

    return result


class SimulatedOCT:
    def __init__(self, laser_sweep_frequency: float = 400e3, detector_sample_frequency: float = 4e9,
                 reference_delay: float = 0.0, dtype=np.complex64):
        self.__reference_delay = reference_delay
        self.__dtype = dtype
        self.__laser_sweep_frequency = laser_sweep_frequency
        self.__detector_sample_frequency = detector_sample_frequency  # Oversample because the source-sweep is non-linear?

        frequency_poly = Polynomial([270665517408362, 1.40171350209608e+19, 2.39878076665655e+25, -4.29710160447565e+31, 3.43649507353042e+37, -1.00660644521354e+43])
        # frequency_poly = Polynomial([270665517408362, 1.9403e+19])  # Uncomment line to simulate a linear sweep instead
        self.__phase = 2 * np.pi * frequency_poly.integ()  # Starts at 0

        self.s_range = ft.Grid(step=1 / self.__detector_sample_frequency, extent=1/self.__laser_sweep_frequency, first=0.0)

        measured_frequencies = frequency_poly(self.s_range)
        bandwidth = np.amax(measured_frequencies) - np.amin(measured_frequencies)
        # mid_spatial_frequency = (np.amin(measured_frequencies) + np.amax(measured_frequencies)) / 2
        dz = 1 / bandwidth
        nb_samples = int(len(self.s_range) * 2 / 3)  # todo: Shortening just for debugging purposes.
        self.__grid = ft.Grid([nb_samples], [dz])  # axial, z

    @property
    def dtype(self):
        """The complex data-type for calculations and output."""
        return self.__dtype

    @property
    def grid(self) -> ft.Grid:
        """The uniformly-spaced spatial grid of the scan volume (or axis)."""
        return self.__grid

    # @property
    # def k_clock(self) -> np.ndarray:
    #     """The 'k-clock' signal as a non-negative cosine between 0 and 1. I.e. the interference of the reference with itself at a fixed delay."""
    #     reference = np.cos(2 * np.pi * self.__spatial_frequency(self.s_range) * self.grid[-1])
    #     autocorrelation = ft.ifft(np.abs(ft.fft(reference)) ** 2).real
    #     # autocorrelation = 0.5 + 0.5 * np.cos(4 * np.pi * self.__spatial_frequency(self.s_range) * self.grid[-1])
    #     return autocorrelation

    @property
    def ground_truth_reflectivity(self) -> np.ndarray:
        """The ground truth reflectivity distribution in space of the sample."""
        sample_reflectivity = np.zeros(self.grid.shape, dtype=self.dtype)
        reflector_separation = 0.5e-3  # Position of first reflector, other two are at 2*reflector_separation and 3*reflector_separation
        d_index = int(reflector_separation / self.grid.step[-1])
        sample_reflectivity[..., d_index::d_index] = 1.0
        return sample_reflectivity

    @property
    def amplitude_ft(self) -> np.ndarray:
        """
        The interferogram or scattering amplitude as a function of spatial frequency in the final dimension (z, or a).

        This calculates the actual image.
        """
        result = np.zeros(self.__grid.shape)
        f = self.__spatial_frequency(self.s_range)[:, np.newaxis]
        fields = np.exp(2j * np.pi * f * self.grid[-1])
        reference_fields = np.exp(2j * np.pi * f * (self.grid[-1] + self.__reference_delay * const.c / 2))
        # sample_fields =
        return result

    @property
    def amplitude(self) -> np.ndarray:
        """The scattering amplitude (field) in real space. Dimensions x, y, z(a)."""
        return ft.ifft(self.amplitude_ft, axis=-1)

    @property
    def intensity(self) -> np.ndarray:
        """The scattering intensity in real space."""
        return np.abs(self.amplitude)


def main():
    # oct = SimulatedOCT()

    # Allow for the frequency of the reference arm to be offset wrt to the sample arm
    
    # Reference arm frequency offset (dictated by phase modulation frequency)
    reference_arm_frequency_offset = 500e6  # Set to 0 for real image
    reflector_separation = 0.5e-3  # Position of first reflector, other two are at 2*reflector_separation and 3*reflector_separation
    
    nb_freq_samples = 640  # Number of frequency sampling points, 640 for 400 kHz source
    
    # This polynomial describes the frequency sweep of the actual source. todo: Is this measured?
    frequency_poly = Polynomial([270665517408362, 1.40171350209608e+19, 2.39878076665655e+25, -4.29710160447565e+31, 3.43649507353042e+37, -1.00660644521354e+43])
    # frequency_poly = Polynomial([270665517408362, 1.9403e+19])  # Uncomment line to simulate a linear sweep instead

    sweep_frequency = 400e3  # 400 kHz source
    sampling_frequency = 4e9  # Sampling frequency should be high enough to minimise numerical error

    sweep_period = 1 / sweep_frequency
    ds = 1 / sampling_frequency  # s = sweep time [s], t = propagation time [s]
    s_range = np.arange(sweep_period / 2 / ds) * ds  # uniform time axis  # todo: why only half sweep period?
    time_between_reflections = reflector_separation * 2 / const.c  # factor of 2 required to account for double-pass

    frequencies_during_scan = frequency_poly(s_range)
    linear_frequency_range = np.linspace(frequencies_during_scan[0], frequencies_during_scan[-1], nb_freq_samples)  # linear frequency vector
    times_at_regular_frequencies = np.interp(linear_frequency_range, frequencies_during_scan, s_range)  # This represents times at which the interferogram is sampled by the k-clock

    wavelengths_during_scan = const.c / frequencies_during_scan
    wavelength_m = np.median(wavelengths_during_scan)
    bandwidth_in_m = np.amax(wavelengths_during_scan) - np.amin(wavelengths_during_scan)
    z_max = wavelength_m ** 2 / bandwidth_in_m * nb_freq_samples / 4  # maximum detectable distance
    
    # frequency_poly_ref describes frequency in reference arm, which is offset by reference_arm_frequency_offset
    frequency_poly_ref = frequency_poly + reference_arm_frequency_offset
    
    # phase of reference arm (we are integrating the frequency sweep to get the phase)
    phase_ref = 2 * np.pi * frequency_poly_ref.integ()(s_range)
    # phases of reflectors in sample arm (same as above)
    phases_sample = [2 * np.pi * frequency_poly.integ()(s_range - (_ + 1) * time_between_reflections) for _ in range(3)]
    # The interferogram is formed by taking the difference between reference arm
    # phase and reflector phase, see Drexler Fujimoto Eqn 2.9 pp.74
    interferogram_during_scan = sum(np.cos(_ - phase_ref) for _ in phases_sample)
    # The interferogram is resampled at fixed frequency intervals
    interferogram_regular = smooth_interp(times_at_regular_frequencies, s_range, interferogram_during_scan)

    # the phase error
    actual_phase_error = smooth_interp(times_at_regular_frequencies, s_range, phase_ref - 2 * np.pi * frequency_poly.integ()(s_range))
    actual_phase_error -= np.linspace(actual_phase_error[0], actual_phase_error[-1], len(actual_phase_error))

    a_scan = np.abs(ft.fftshift(ft.ifft(interferogram_regular)))

    log.info(f'Computing spectrogram from interferogram of shape {interferogram_regular.shape} using stft...')
    # STFT, window size is 1/4 of interferogram, which is then shifted by one point,
    # each windowed interferogram is zero-padded to 2*N prior to fft
    f_range_spectrogram, t_range_spectrogram, spectrogram = stft(interferogram_regular, sampling_frequency,
                                                                 window=hann(nb_freq_samples // 4), return_onesided=False,
                                                                 noverlap=(nb_freq_samples // 4) - 1, nfft=4 * nb_freq_samples,
                                                                 nperseg=nb_freq_samples // 4)

    log.info(f'Computed spectrogram of shape {spectrogram.shape} using stft.')
    
    z_range = np.linspace(-z_max, z_max, spectrogram.shape[0])
    a_scan_z_range = np.linspace(-z_max, z_max, len(a_scan))

    log.info('Displaying...')
    z_range_plotlims = np.array([-1, 1]) * z_max
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].pcolor(t_range_spectrogram / 1e-6, z_range / 1e-3, np.abs(spectrogram))
    axs[0, 0].hlines(0, t_range_spectrogram[0] / 1e-6, t_range_spectrogram[-1] / 1e-6, colors='r', linestyles='--')
    axs[0, 0].set(#xlim=[0, np.amax(t_range_spectrogram) / 1e-6], #ylim=z_range_plotlims / 1e-3,
                  xlabel=r'Time / $\mu$s', ylabel='Distance / mm',
                  title='A-scan: STFT of Interferogram')
    axs[0, 1].plot(a_scan_z_range / 1e-3, a_scan, 'r')
    axs[0, 1].set(#xlim=z_range_plotlims / 1e-3,
                  xlabel='Distance / mm', ylabel='Amplitude / a.u.',
                  title='A-scan: Effects of Resampling')

    log.info('Correcting for modulation-induced dispersion using xcorr STFT method...')
    # Due to complex conjugate, only half of the spectrogram
    half_spectrogram = spectrogram[:spectrogram.shape[0] // 2]
    # Choosing the centre point of the STFT as the reference
    spectrogram_reference = half_spectrogram[:, half_spectrogram.shape[1] // 2, np.newaxis]

    log.info(f'Computing normalized cross correlation of a spectrogram with shape {half_spectrogram.shape} with its its central spectrum...')
    # Xcorr of STFT
    # In theory we could get the dispersion function directly from the STFT,
    # but this method should be more robust in the presence of a complicated sample
    correlation = normxcorr2(np.abs(spectrogram_reference), np.abs(half_spectrogram))
    log.info(f'Computed normalized cross correlation of shape {correlation.shape} of the spectrogram.')
    
    # frequency scale for xcorr, it involves only the positive frequencies but in twice as many points
    f_range_correlation = np.interp(np.linspace(0, 1, len(f_range_spectrogram)),
                                    np.linspace(0, 1, len(f_range_spectrogram[f_range_spectrogram.size//2:])), f_range_spectrogram[f_range_spectrogram.size//2:])
    
    # Extract ridge from xcorr, this tells us how the a-scan as a whole "moves" over the course of the sweep
    f_ridge = tfridge(correlation, f_range_correlation, 1e-2)

    # fig, axs = plt.subplots(2, 1)
    # from optics.utils.display import complex2rgb
    # axs[0].imshow(complex2rgb(spectrogram, 1))
    # axs[1].imshow(correlation)
    # print(half_spectrogram.shape)
    # plt.show()
    # exit()

    # Need to interpolate to match the size of the vector to the interferogram
    f_ridge_interp = np.interp(np.linspace(0, 1, len(interferogram_regular)),
                               np.linspace(0, 1, len(f_ridge)), f_ridge)
    
    # Corrective phase is the integral of the STFT ridge
    fridge_phase = np.cumsum(f_ridge_interp) / sampling_frequency
    
    # Linear phase shifts the image and does not affect resolution, 
    f_ridge_phase_nonlinear = fridge_phase - np.linspace(fridge_phase[0], fridge_phase[-1], len(fridge_phase))

    f_ridge_phase_nonlinear *= 18  # todo: Dorian: Not sure why an extra factor is needed here, I think my frequency axis might be wrong in the tfridge function call above
    phasor = np.exp(1j * f_ridge_phase_nonlinear)
    
    # applying phase correction
    corrected_interferogram = interferogram_regular * phasor
    corrected_a_scan = np.abs(ft.fftshift(ft.ifft(corrected_interferogram)))

    f_range_plotlims = np.array([np.amin(f_range_spectrogram), np.amax(f_range_spectrogram)])
    axs[0, 2].pcolor(t_range_spectrogram / 1e-6, f_range_spectrogram[:, np.newaxis] / 1e9, correlation[:, :-1])
    axs[0, 2].plot(t_range_spectrogram / 1e-6, f_ridge / 1e9, 'r')
    axs[0, 2].set(#xlim=[0, np.amax(t_range_spectrogram) / 1e-6],  # ylim=f_range_plotlims / 1e9,
                  xlabel=r'Time / $\mu$s', ylabel='Frequency / GHz', title='Cross-correlation of stft')

    log.info(f'Computing corrected spectrogram from corrected interferogram of shape {corrected_interferogram.shape} using stft...')
    f_range_spectrogram, t_range_spectrogram, corrected_spectrogram = stft(corrected_interferogram, sampling_frequency,
                                                                           window=hann(nb_freq_samples // 4), return_onesided=False,
                                                                           noverlap=(nb_freq_samples // 4) - 1,
                                                                           nfft=2 * nb_freq_samples, nperseg=nb_freq_samples // 4)  # todo: why is nfft length only 2* not 4*?
    log.info(f'Computed corrected spectrogram of shape {corrected_spectrogram.shape} using stft.')

    axs[1, 0].pcolor(t_range_spectrogram / 1e-6, ft.ifftshift(f_range_spectrogram), ft.ifftshift(np.abs(corrected_spectrogram) ** 2, axes=0))
    axs[1, 0].set(xlabel=r'time / $\mu$s', ylabel='frequency', title='corrected interferogram intensity')
    axs[1, 1].plot(a_scan_z_range, corrected_a_scan)
    axs[1, 1].set(xlabel='a-scan z', ylabel='Corrected A-scan')
    axs[1, 2].axis('off')

    log.info('Correcting for modulation-induced dispersion using phase difference...')
    # The below method seems to produce near ideal phase correction, but not really implementable in practice
    
    ideal_interferogram = interferogram_regular * np.exp(1j * actual_phase_error)

    log.info(f'Computing ideal spectrogram from ideal interferogram of shape {ideal_interferogram.shape} using stft...')
    f, f_range_spectrogram, ideal_spectrogram = stft(ideal_interferogram, sampling_frequency,
                                                     window=hann(nb_freq_samples // 4), return_onesided=False,
                                                     noverlap=(nb_freq_samples // 4) - 1,
                                                     nfft=2 * nb_freq_samples, nperseg=nb_freq_samples // 4)
    log.info(f'Computed ideal spectrogram of shape {ideal_spectrogram.shape} using stft.')

    axs[2, 0].pcolor(f_range_spectrogram / 1e-6, ft.ifftshift(f), ft.ifftshift(np.abs(ideal_spectrogram) ** 2, axes=0))
    axs[2, 0].set(xlabel=r'time / $\mu$s', ylabel='frequency', title='ideal interferogram intensity')
    axs[2, 1].plot(f_range_spectrogram[1:] / 1e-6, np.abs(ft.fftshift(ft.ifft(ideal_interferogram))))
    axs[2, 1].set(xlabel=r'time / $\mu$s', ylabel='ideal a-scan')
    log.info('Plotting corrective phase from STFT and actual phase error...')
    axs[2, 2].plot(f_range_spectrogram[1:] / 1e-6, np.unwrap(f_ridge_phase_nonlinear), label='estimated phase error')
    axs[2, 2].plot(f_range_spectrogram[1:] / 1e-6, np.unwrap(actual_phase_error), label='actual phase error')
    axs[2, 2].set(xlabel=r'time / $\mu$s', ylabel=r'$\phi$  / rad')
    axs[2, 2].legend()

    log.info('Done!')
    plt.show()


if __name__ == '__main__':
    main()
