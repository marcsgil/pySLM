import numpy as np
from inspect import signature
import logging

log = logging.getLogger(__name__)

from optics.instruments.slm import SLM
from optics.utils.ft.grid import Grid


def measure_aberration(slm: SLM, measurement_function, measurement_grid_shape=(15, 15, 3),
                       progress_callback=lambda fraction, pupil: True) -> np.ndarray:
    """
    Determines the aberration based on interference of different deflections
    and calculates the correction function. Multiple probes can be used, such
    as individual pixels of a camera, or NSOM tip. See
    camAberrationMeasurement.m for an example. When multiple probes are
    given, the aberrations are estimated simultaneously for each probe.

    :param slm: The spatial light modulator to use.
    :param measurement_function: a function that returns the probe value, it optionally takes as argument the complex field at each pixel of the SLM.
    :param measurement_grid_shape:  A vector of three scalars, the number of probes in y and x followed by the number of phase probes (must be >=3).
    :param progress_callback: When specified, this function will be executed every spatial frequency probe, it takes as optional arguments:
                 - fraction_done: the completion ratio
                 - current_pupil_function_estimate: the complex matrix with the current estimate of the pupil function

    :return: measured_pupil_function: the complex matrix with the estimate of the aberrated pupil function
    """
    if len(signature(measurement_function).parameters) < 1:
        def unified_measurement_function(complex_field: np.ndarray):
            return measurement_function()
    else:
        unified_measurement_function = measurement_function

    measurement_grid_shape = np.array(measurement_grid_shape)
    if measurement_grid_shape.size < 2:
        measurement_grid_shape[1] = 0
        log.info('The horizontal deflection has not been specified, defaulting to 0 (vertical only deflection).')
    if measurement_grid_shape.size < 3:
        measurement_grid_shape[2] = 3
        log.info(f'The number of phases to sample has not been specified, defaulting to {measurement_grid_shape[2]}')

    # Define the measurement modes
    sample_grid = Grid(shape=measurement_grid_shape[:2], step=1.0 / slm.roi.shape)
    radius_squared = sample_grid[0]**2 + sample_grid[1]**2
    sorted_indexes = np.argsort(radius_squared.ravel())

    def get_mode(idx):
        unraveled_coordinates = np.unravel_index(sorted_indexes[idx], sample_grid.shape)
        # print(f'{idx}, {sorted_indexes[idx]}: {unraveled_coordinates}')
        if idx > 0:
            return np.exp(2j * np.pi * (slm.grid[0] * sample_grid[0].ravel()[unraveled_coordinates[0]] +
                                        slm.grid[1] * sample_grid[1].ravel()[unraveled_coordinates[1]]))
        else:
            return 1  # just for efficiency

    reference = get_mode(0)  # Use the first mode as the reference

    # Half the field in the reference beam
    # todo: in principle this could be dynamically adapted
    reference_fraction = 0.5

    # Probe for different deflections
    current_pupil_function_estimate = np.zeros(slm.roi.shape, dtype=complex)
    measured_coefficients = np.zeros(measurement_grid_shape[:2], dtype=complex)
    for measurement_idx in range(sorted_indexes.size):
        # Determine the mode to measure with in this iteration
        complex_mode = get_mode(measurement_idx)

        # Test different phase offsets
        nb_of_phase_measurements = measurement_grid_shape[2]
        intensity_at_phase = np.zeros(nb_of_phase_measurements)
        test_phasors = np.exp(2j * np.pi * np.arange(nb_of_phase_measurements) / nb_of_phase_measurements)
        for phase_idx, phasor in enumerate(test_phasors):
            interference = reference_fraction * reference + (1 - reference_fraction) * phasor * complex_mode
            # Modulate the SLM
            slm.modulate(interference)
            # Measure the target function
            intensity_at_phase[phase_idx] = unified_measurement_function(interference)

        reference_fraction_correction = 1 / (reference_fraction * (1 - reference_fraction))

        # Work out the phase and amplitude from the sampling using 'lock-in' amplification
        measured_coefficient = reference_fraction_correction * np.vdot(test_phasors, intensity_at_phase)  # complex conjugation of first argument of vdot()

        # Verify and store the reference, this is always calculated first
        if measurement_idx == 0:
            # The first measurement is a self-reference test, so the result should be a positive real number
            measurement_error_estimate = np.abs(np.angle(measured_coefficient))
            if measurement_error_estimate > 0.01:
                log.info(f'The estimated measurement error is large:  {measurement_error_estimate*100:0.2f}%')

            # Force to be real, the imaginary part must be a measurement error for the reference
            measured_coefficient = np.real(measured_coefficient)

        # Store the measurement multidimensional
        measured_coefficients.ravel()[measurement_idx] = measured_coefficient
        current_pupil_function_estimate += measured_coefficient * np.conj(complex_mode)
        # Report progress
        cont = progress_callback((measurement_idx+1) / sorted_indexes.size, current_pupil_function_estimate)
        if cont is not None and not cont:
            log.info("The user-specified progress callback returned False to indicate an early exit. "
                     + "Stopping measurement now.")
            break

    # Un-sort the coefficients
    measured_coefficients.ravel()[sorted_indexes] = measured_coefficients.ravel()
    # Check
    from optics.utils import ft
    current_pupil_function_estimate_ft = ft.fftshift(ft.fft2(ft.ifftshift(current_pupil_function_estimate)))
    current_pupil_function_estimate_ft = current_pupil_function_estimate_ft[
                                         current_pupil_function_estimate_ft.shape[0]//2-1:current_pupil_function_estimate_ft.shape[0]//2+2,
                                         current_pupil_function_estimate_ft.shape[1]//2-1:current_pupil_function_estimate_ft.shape[1]//2+2]
    # log.info(np.mean(np.angle(measured_coefficients / current_pupil_function_estimate_ft)))
    # log.info(np.mean(np.angle(measured_coefficients.conj() / current_pupil_function_estimate_ft)))
    # log.info(np.mean(measured_coefficients / current_pupil_function_estimate_ft))

    return current_pupil_function_estimate
